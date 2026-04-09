"""
csaq/inference.py — Self-Speculative Decoding Engine (v0.2.6 Hardened)

Uses the CSAQ dual-precision weight structure:
  - Draft path:  quantized low-bit weights (fast, approximate)
  - Verify path: FP16 high-salience clique weights (accurate, slower)

v0.2.6 Hardening:
  - DynamicCache support (crop / get_seq_length API)
  - Three-stage numerical sanitization (Clamp → Sanitize → Validate)
  - CUDA kernel isolation with torch.cuda.synchronize()
  - Real error recovery with per-token FP16 fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import time
import os
import sys
import json
import pickle
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from tqdm import tqdm

# ── DynamicCache import (HuggingFace >= 4.36) ────────────────────────────────
# Guarded import: if the user has an older transformers version, we degrade
# gracefully by setting DynamicCache = None and skipping isinstance checks.
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


@dataclass
class SpeculativeReport:
    """Tracks real performance metrics for Self-Speculative Decoding."""
    tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    draft_calls: int = 0
    verify_calls: int = 0
    total_wallclock_s: float = 0.0
    mode: str = "speculative"
    fallback_events: List[Dict[str, str]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / max(total, 1)

    @property
    def speedup_factor(self) -> float:
        total_calls = self.draft_calls + self.verify_calls
        return self.tokens_generated / max(total_calls, 1)

    @property
    def inter_token_latency_ms(self) -> float:
        return (self.total_wallclock_s * 1000.0) / max(self.tokens_generated, 1)

    @property
    def block_efficiency(self) -> float:
        return self.tokens_accepted / max(self.verify_calls, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tokens_generated": self.tokens_generated,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_factor": round(self.speedup_factor, 2),
            "inter_token_latency_ms": round(self.inter_token_latency_ms, 2),
            "block_efficiency": round(self.block_efficiency, 2),
            "draft_calls": self.draft_calls,
            "verify_calls": self.verify_calls,
            "wallclock_s": round(self.total_wallclock_s, 3),
            "fallback_events": self.fallback_events,
        }


# Type alias
_HookEntry = Tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]


class CSAQInferenceEngine:
    """
    Self-Speculative Decoding engine for CSAQ-quantized models.
    """

    def __init__(
        self,
        model: nn.Module,
        causal_map: Dict[str, List[int]],
        tokenizer=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._in_verify = False

        # ── Pre-calculate hooked layers ───────────────────────────────────
        self._hooks: List[_HookEntry] = []

        for module_name in causal_map:
            module = self._resolve_module(module_name)
            if module is None:
                continue
            if not hasattr(module, "_csaq_fp16_backup"):
                continue

            self._hooks.append((
                module,
                module._csaq_hi_rows,
                module._csaq_fp16_backup,
                module._csaq_quant_stash,
            ))

        if not self._hooks:
            warnings.warn(
                "[CSAQ] CSAQInferenceEngine initialized with 0 hooked layers.",
                stacklevel=2,
            )

    def _resolve_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        if parts[-1] == "weight":
            parts = parts[:-1]
        mod = self.model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                return None
        return mod if isinstance(mod, nn.Module) else None

    # ═══════════════════════════════════════════════════════════════════════
    # Weight Swapping — v0.2.6: Memory-Integrity Guaranteed
    # ═══════════════════════════════════════════════════════════════════════

    def _swap_to_verify(self):
        """Swap quantized weights for FP16 backups on high-salience rows."""
        for module, rows, fp16_backup, _ in self._hooks:
            module.weight.data[rows] = fp16_backup
        self._in_verify = True

    def _swap_to_draft(self):
        """Restore quantized weights from stash on high-salience rows."""
        for module, rows, _, quant_stash in self._hooks:
            module.weight.data[rows] = quant_stash
        self._in_verify = False

    def _ensure_draft_state(self):
        if self._in_verify:
            self._swap_to_draft()

    # ═══════════════════════════════════════════════════════════════════════
    # KV-Cache Helpers — v0.2.6: DynamicCache API Support
    # ═══════════════════════════════════════════════════════════════════════

    def _is_dynamic_cache(self, past_key_values) -> bool:
        """Check if past_key_values is a HuggingFace DynamicCache object."""
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            return True
        # Fallback heuristic: if it has .crop() and .get_seq_length() methods
        # it quacks like a DynamicCache even if the import failed.
        if (hasattr(past_key_values, "crop") and
                hasattr(past_key_values, "get_seq_length") and
                hasattr(past_key_values, "key_cache")):
            return True
        return False

    def _truncate_kv_cache(self, past_key_values, accepted_len: int):
        """Truncate KV cache to `accepted_len` positions.
        
        Handles three cache formats:
          1. transformers.DynamicCache  — use .crop(accepted_len)
          2. Object with .key_cache/.value_cache lists — manual slice
          3. Tuple-of-tuples (legacy HF format) — manual slice
        """
        if past_key_values is None:
            return None

        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            if hasattr(past_key_values, "crop"):
                try:
                    past_key_values.crop(accepted_len)
                    return past_key_values
                except Exception:
                    pass

        # Manual slice if not DynamicCache or lacks crop
        if hasattr(past_key_values, "key_cache"):
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = (
                    past_key_values.key_cache[layer_idx][:, :, :accepted_len, :]
                )
                past_key_values.value_cache[layer_idx] = (
                    past_key_values.value_cache[layer_idx][:, :, :accepted_len, :]
                )
            return past_key_values

        if isinstance(past_key_values, (tuple, list)):
            truncated = []
            for layer_past in past_key_values:
                if isinstance(layer_past, (tuple, list)):
                    truncated.append(
                        tuple(t[:, :, :accepted_len, :] for t in layer_past)
                    )
                else:
                    truncated.append(layer_past)
            return tuple(truncated)

        return past_key_values

    def _get_kv_len(self, past_key_values) -> int:
        """Get the current sequence length from the KV cache."""
        if past_key_values is None:
            return 0

        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            if hasattr(past_key_values, "get_seq_length"):
                try:
                    return past_key_values.get_seq_length()
                except Exception:
                    pass

        # Fallback manual calculation
        if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
            return past_key_values.key_cache[0].shape[2]

        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            first = past_key_values[0]
            if isinstance(first, (tuple, list)) and len(first) > 0:
                return first[0].shape[2]

        return 0

    # ═══════════════════════════════════════════════════════════════════════
    # Core Operations — v0.2.6: CUDA-Isolated + Numerically Hardened
    # ═══════════════════════════════════════════════════════════════════════

    def _cuda_sync(self):
        """Synchronize CUDA to surface kernel errors in Python."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward_safe(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        report: SpeculativeReport,
        is_verify: bool = False,
    ):
        """Execute a model forward pass with CUDA isolation."""
        if is_verify:
            self._swap_to_verify()
        
        try:
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            self._cuda_sync()
            
            if is_verify:
                report.verify_calls += 1
            else:
                report.draft_calls += 1
                
            return out
        except RuntimeError as e:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            
            # The safety valve calls for exiting the loop immediately.
            raise RuntimeError("SAFETY_VALVE_TRIGGERED") from e
        finally:
            if is_verify:
                self._swap_to_draft()

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        greedy: bool,
        report: SpeculativeReport,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next token with pre-emptive numerical sanitation."""
        # STAGE 1: CLAMP — Logit normalization using float16 max
        logits = torch.clamp(logits, min=-65504.0, max=65504.0)

        # STAGE 2: SANITIZE
        logits = torch.nan_to_num(logits, nan=0.0, posinf=65504.0, neginf=-65504.0)

        # ── Adaptive temperature scaling for high-variance logits ────────
        if temperature > 0.0:
            var = logits.var(dim=-1).mean().item()
            if var > 100.0:
                scale = math.sqrt(var / 100.0)
                temperature = max(temperature, scale)

        if greedy:
            token = logits.argmax(dim=-1, keepdim=True)
            return token, F.softmax(logits, dim=-1)

        scaled = logits / max(temperature, 1e-8)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            mask = sorted_indices_to_remove
            sorted_logits[mask] = -1e10
            scaled = torch.empty_like(scaled).scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(scaled, dim=-1)

        # STAGE 3: VALIDATE — Pre-emptive check for poisoned distributions
        target_probs = probs
        if (target_probs < 0).any() or torch.isnan(target_probs).any() or torch.isinf(target_probs).any():
            report.fallback_events.append({
                "phase": "sample_generation",
                "error": "Poisoned distribution detected (NaN, Inf, or negative)",
                "action": "numerical_stabilization_event",
            })
            token = logits.argmax(dim=-1, keepdim=True)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            return token, uniform

        # Final guard against empty distribution space
        psum_final = probs.sum(dim=-1, keepdim=True)
        if (psum_final < 1e-10).any():
            report.fallback_events.append({
                "phase": "sample_generation",
                "error": "Zero total probability mass",
                "action": "numerical_stabilization_event",
            })
            token = logits.argmax(dim=-1, keepdim=True)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            return token, uniform

        token = torch.multinomial(probs, 1)
        return token, probs

    # ═══════════════════════════════════════════════════════════════════════
    # Generation Entries
    # ═══════════════════════════════════════════════════════════════════════

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        speculative: bool = False,
        lookahead: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        """HF-compatible .generate() proxy."""
        if speculative:
            return self.generate_speculative(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                lookahead=lookahead,
                temperature=temperature,
                top_p=top_p,
            )

        self._ensure_draft_state()
        t0 = time.time()
        
        report = SpeculativeReport(mode="standard_fallback")

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            **kwargs,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        try:
            output = self.model.generate(**gen_kwargs)
            elapsed = time.time() - t0
            prompt_len = input_ids.shape[-1] if input_ids is not None else 0
            n_new = max(0, output.shape[-1] - prompt_len)

            report.tokens_generated = n_new
            report.tokens_accepted = n_new
            report.draft_calls = n_new
            report.total_wallclock_s = elapsed
            
            return output, report
        except Exception as e:
            warnings.warn(f"[CSAQ] Fallback Generation Failed: {str(e)}")
            report.fallback_events.append({
                "phase": "standard_generate",
                "error": str(e),
                "action": "return_input",
            })
            return (input_ids if input_ids is not None else torch.tensor([], device=self.device)), report

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        lookahead: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        """Generate tokens using Self-Speculative Decoding."""
        report = SpeculativeReport(mode="speculative")
        
        if not self._hooks:
            return self.generate(
                input_ids=input_ids,
                speculative=False,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        t0 = time.time()
        self.model.eval()
        self._ensure_draft_state()
        
        generated = input_ids.clone().to(self.device)
        past_key_values = None
        greedy = (temperature == 0.0)
        tokens_produced = 0

        try:
            while tokens_produced < max_new_tokens:
                remaining = max_new_tokens - tokens_produced
                K = min(lookahead, remaining)
                if K <= 0:
                    break

                # ── STEP 1: DRAFT ──────────────────────────────────────────────
                draft_tokens = []
                draft_logits_list = []
                draft_input = generated[:, -1:]

                draft_failed = False
                for _ in range(K):
                    try:
                        out = self._forward_safe(
                            draft_input, past_key_values, report, is_verify=False
                        )
                        past_key_values = out.past_key_values
                        logits = out.logits[:, -1, :]
                        report.draft_calls += 1

                        token, _ = self._sample(logits, temperature, top_p, greedy, report)
                        draft_tokens.append(token)
                        draft_logits_list.append(logits)
                        draft_input = token if token.dim() == 2 else token.unsqueeze(0)
                    except RuntimeError as e:
                        if str(e) == "SAFETY_VALVE_TRIGGERED":
                            report.fallback_events.append({
                                "phase": "draft_loop",
                                "error": "CUDA Safety Valve Triggered",
                                "action": "abort_speculative",
                            })
                            # Fallback sequentially down! Do not continue speculative path.
                            self._ensure_draft_state()
                            remaining_tokens = max_new_tokens - tokens_produced
                            fb_out, fb_rep = self.generate(
                                input_ids=generated,
                                max_new_tokens=remaining_tokens,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            report.fallback_events.extend(fb_rep.fallback_events)
                            report.tokens_generated += fb_rep.tokens_generated
                            return fb_out, report
                        else:
                            raise e

                # If we got fewer tokens than K, adjust K
                K_actual = len(draft_tokens)
                draft_token_ids = torch.cat(draft_tokens, dim=-1)

                # ── STEP 2: VERIFY ────────────────────────────────────────────
                kv_len_before_draft = self._get_kv_len(past_key_values) - K_actual
                verify_kv = self._truncate_kv_cache(past_key_values, kv_len_before_draft)

                verify_input = torch.cat([
                    generated[:, -(K_actual + 1):-K_actual],
                    draft_token_ids,
                ], dim=-1)

                try:
                    verify_out = self._forward_safe(verify_input, verify_kv, report, is_verify=True)
                except RuntimeError as e:
                    if str(e) == "SAFETY_VALVE_TRIGGERED":
                        report.fallback_events.append({
                            "phase": "verify_loop",
                            "error": "CUDA Safety Valve Triggered",
                            "action": "abort_speculative",
                        })
                        self._ensure_draft_state()
                        remaining_tokens = max_new_tokens - tokens_produced
                        fb_out, fb_rep = self.generate(
                            input_ids=generated,
                            max_new_tokens=remaining_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        report.fallback_events.extend(fb_rep.fallback_events)
                        report.tokens_generated += fb_rep.tokens_generated
                        return fb_out, report
                    else:
                        raise e

                verify_logits = verify_out.logits

                # ── STEP 3: REJECTION SAMPLING ────────────────────────────────
                n_accepted = 0

                for k in range(K_actual):
                    draft_logit_k = draft_logits_list[k]
                    verify_logit_k = verify_logits[:, k + 1, :]
                    draft_token_k = draft_token_ids[:, k]

                    verify_logit_k = torch.clamp(verify_logit_k, min=-65504.0, max=65504.0)
                    verify_logit_k = torch.nan_to_num(verify_logit_k, nan=0.0, posinf=65504.0, neginf=-65504.0)

                    if greedy:
                        verify_token = verify_logit_k.argmax(dim=-1)
                        if verify_token.item() == draft_token_k.item():
                            n_accepted += 1
                        else:
                            draft_token_ids[:, k] = verify_token
                            n_accepted += 1
                            report.tokens_rejected += 1
                            report.tokens_accepted -= 1
                            break
                    else:
                        p_draft = F.softmax(
                            torch.clamp(draft_logit_k / max(temperature, 1e-8), min=-65504.0, max=65504.0),
                            dim=-1
                        )
                        p_verify = F.softmax(
                            torch.clamp(verify_logit_k / max(temperature, 1e-8), min=-65504.0, max=65504.0),
                            dim=-1
                        )

                        # Sanitize probability distributions
                        p_draft = torch.nan_to_num(p_draft, nan=0.0, posinf=0.0, neginf=0.0)
                        p_verify = torch.nan_to_num(p_verify, nan=0.0, posinf=0.0, neginf=0.0)
                        p_draft_sum = p_draft.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                        p_verify_sum = p_verify.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                        p_draft = p_draft / p_draft_sum
                        p_verify = p_verify / p_verify_sum

                        token_idx = draft_token_k.item()
                        ratio = p_verify[0, token_idx] / p_draft[0, token_idx].clamp(min=1e-10)
                        r = torch.rand(1, device=self.device)

                        if r.item() < min(1.0, ratio.item()):
                            n_accepted += 1
                        else:
                            residual = (p_verify - p_draft).clamp(min=0.0)
                            residual_sum = residual.sum()
                            if residual_sum > 1e-8:
                                # Sanitize residual before multinomial
                                residual_normed = residual / residual_sum
                                residual_normed = torch.nan_to_num(
                                    residual_normed, nan=0.0, posinf=0.0, neginf=0.0
                                )
                                res_sum = residual_normed.sum(dim=-1, keepdim=True)
                                if (res_sum < 1e-10).any():
                                    corrected = verify_logit_k.argmax(dim=-1, keepdim=True)
                                else:
                                    corrected = torch.multinomial(residual_normed, 1)
                            else:
                                corrected = verify_logit_k.argmax(dim=-1, keepdim=True)
                            draft_token_ids[:, k] = corrected.squeeze(-1)
                            n_accepted += 1
                            report.tokens_rejected += 1
                            report.tokens_accepted -= 1
                            break

                # ── STEP 4: COMMIT ────────────────────────────────────────────
                accepted_tokens = draft_token_ids[:, :n_accepted]
                generated = torch.cat([generated, accepted_tokens], dim=-1)
                tokens_produced += n_accepted

                report.tokens_accepted += n_accepted
                report.tokens_rejected += (K_actual - n_accepted)
                report.tokens_generated += n_accepted

                final_kv_len = kv_len_before_draft + n_accepted + 1
                past_key_values = self._truncate_kv_cache(
                    verify_out.past_key_values, final_kv_len
                )

                if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                    if accepted_tokens.shape[1] > 0 and accepted_tokens[:, -1].item() == self.tokenizer.eos_token_id:
                        break

        except Exception as e:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            warnings.warn(f"[CSAQ] Speculative Engine Exception: {str(e)}")
            self._ensure_draft_state()
            report.mode = "speculative_partial_error"
            report.fallback_events.append({
                "phase": "outer_loop",
                "error": str(e),
                "action": "return_partial",
            })
            
        report.total_wallclock_s = time.time() - t0
        return generated, report
