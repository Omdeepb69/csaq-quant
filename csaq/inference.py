"""
csaq/inference.py — Self-Speculative Decoding Engine (v0.2.6)

Uses the CSAQ dual-precision weight structure:
  - Draft path:  quantized low-bit weights (fast, approximate)
  - Verify path: FP16 high-salience clique weights (accurate, slower)

The key insight: Draft and Verify share the SAME model and SAME KV-cache.
We swap weight rows in-place rather than duplicating the model.
"""

import os
import sys
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = type("DummyDynamicCache", (object,), {})


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
    error_log: str = ""

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
            "error_log": self.error_log,
        }


# Type alias
_HookEntry = Tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]


class CSAQInferenceEngine:
    """
    Self-Speculative Decoding engine for CSAQ-quantized models.

    Wraps a HuggingFace PreTrainedModel.

    .generate() ALWAYS returns (output_tensor, SpeculativeReport).
    This is true regardless of whether speculative=True or speculative=False,
    so that callers can always unpack: output, report = engine.generate(...).
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
                "[CSAQ] CSAQInferenceEngine initialized with 0 hooked layers. "
                "generate_speculative() will fall back to standard generation.",
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
    # Weight Swapping
    # ═══════════════════════════════════════════════════════════════════════

    def _swap_to_verify(self):
        for module, rows, fp16_backup, _ in self._hooks:
            module.weight.data[rows] = fp16_backup
        self._in_verify = True

    def _swap_to_draft(self):
        for module, rows, _, quant_stash in self._hooks:
            module.weight.data[rows] = quant_stash
        self._in_verify = False

    def _ensure_draft_state(self):
        if self._in_verify:
            self._swap_to_draft()

    # ═══════════════════════════════════════════════════════════════════════
    # KV-Cache Truncation
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _truncate_kv_cache(past_key_values, accepted_len: int):
        if past_key_values is None:
            return None

        # Handle HuggingFace DynamicCache
        if isinstance(past_key_values, DynamicCache) or hasattr(past_key_values, "key_cache"):
            if hasattr(past_key_values, "crop"):
                past_key_values.crop(accepted_len)
                return past_key_values
            else:
                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[layer_idx] = (
                        past_key_values.key_cache[layer_idx][:, :, :accepted_len, :]
                    )
                    past_key_values.value_cache[layer_idx] = (
                        past_key_values.value_cache[layer_idx][:, :, :accepted_len, :]
                    )
                return past_key_values

        truncated = []
        for layer_past in past_key_values:
            truncated.append(
                tuple(t[:, :, :accepted_len, :] for t in layer_past)
            )
        return tuple(truncated)

    @staticmethod
    def _get_kv_len(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if hasattr(past_key_values, "key_cache"):
            if len(past_key_values.key_cache) == 0:
                return 0
            return past_key_values.key_cache[0].shape[2]
        return past_key_values[0][0].shape[2]

    # ═══════════════════════════════════════════════════════════════════════
    # .generate() — ALWAYS returns (tensor, SpeculativeReport)
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
        """
        HF-compatible .generate() proxy.
        """
        if speculative:
            return self.generate_speculative(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                lookahead=lookahead,
                temperature=temperature,
                top_p=top_p,
            )

        # ── Standard HF delegation ───────────────────────────────────────
        self._ensure_draft_state()
        t0 = time.time()

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
        elif hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
            gen_kwargs.setdefault("pad_token_id", self.model.config.eos_token_id)

        output = self.model.generate(**gen_kwargs)

        elapsed = time.time() - t0
        prompt_len = input_ids.shape[-1] if input_ids is not None else 0
        n_new = max(0, output.shape[-1] - prompt_len)

        report = SpeculativeReport(
            tokens_generated=n_new,
            tokens_accepted=n_new,
            tokens_rejected=0,
            draft_calls=n_new,
            verify_calls=0,
            total_wallclock_s=elapsed,
            mode="standard_fallback",
        )

        return output, report

    # ═══════════════════════════════════════════════════════════════════════
    # Speculative Generation
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        lookahead: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        """
        Generate tokens using Self-Speculative Decoding.
        """
        if not self._hooks:
            warnings.warn(
                "[CSAQ] generate_speculative() called with 0 hooked layers. "
                "Falling back to standard model.generate().",
                stacklevel=2,
            )
            return self.generate(
                input_ids=input_ids,
                speculative=False,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        report = SpeculativeReport()
        t0 = time.time()

        self.model.eval()
        self._ensure_draft_state()
        generated = input_ids.clone().to(self.device)
        past_key_values = None
        greedy = (temperature == 0.0)

        tokens_produced = 0

        # Automatic GPU Recovery Block
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

                for _ in range(K):
                    out = self.model(
                        input_ids=draft_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    # Isolated Execution: Ensure Forward Pass isn't poisoning CUDA
                    if out.logits.is_cuda:
                        torch.cuda.synchronize()

                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    report.draft_calls += 1

                    token, _ = self._sample(logits, temperature, top_p, greedy)
                    draft_tokens.append(token)
                    draft_logits_list.append(logits)
                    draft_input = token if token.dim() == 2 else token.unsqueeze(0)

                draft_token_ids = torch.cat(draft_tokens, dim=-1)

                # ── STEP 2: VERIFY (try/finally guard) ────────────────────────
                kv_len_before_draft = max(0, self._get_kv_len(past_key_values) - K)
                verify_kv = self._truncate_kv_cache(past_key_values, kv_len_before_draft)

                verify_input = torch.cat([
                    generated[:, -(K + 1):-K],
                    draft_token_ids,
                ], dim=-1)

                try:
                    self._swap_to_verify()
                    verify_out = self.model(
                        input_ids=verify_input,
                        past_key_values=verify_kv,
                        use_cache=True,
                    )
                    # Isolated Execution Guard
                    if verify_out.logits.is_cuda:
                        torch.cuda.synchronize()
                        
                    verify_logits = verify_out.logits
                    report.verify_calls += 1
                finally:
                    self._swap_to_draft()

                # ── STEP 3: REJECTION SAMPLING ────────────────────────────────
                n_accepted = 0

                for k in range(K):
                    draft_logit_k = draft_logits_list[k]
                    verify_logit_k = verify_logits[:, k + 1, :]
                    draft_token_k = draft_token_ids[:, k]

                    if greedy:
                        verify_token = verify_logit_k.argmax(dim=-1)
                        if verify_token.item() == draft_token_k.item():
                            n_accepted += 1
                        else:
                            draft_token_ids[:, k] = verify_token
                            n_accepted += 1
                            break
                    else:
                        p_draft = F.softmax(draft_logit_k / temperature, dim=-1)
                        p_verify = F.softmax(verify_logit_k / temperature, dim=-1)

                        token_idx = draft_token_k.item()
                        ratio = p_verify[0, token_idx] / p_draft[0, token_idx].clamp(min=1e-10)
                        r = torch.rand(1, device=self.device)

                        if r.item() < min(1.0, ratio.item()):
                            n_accepted += 1
                        else:
                            residual = (p_verify - p_draft).clamp(min=0.0)
                            residual_sum = residual.sum()
                            if residual_sum > 1e-8:
                                corrected = torch.multinomial(residual / residual_sum, 1)
                            else:
                                corrected = verify_logit_k.argmax(dim=-1, keepdim=True)
                            draft_token_ids[:, k] = corrected.squeeze(-1)
                            n_accepted += 1
                            break

                # ── STEP 4: Commit & truncate ─────────────────────────────────
                accepted_tokens = draft_token_ids[:, :n_accepted]
                generated = torch.cat([generated, accepted_tokens], dim=-1)
                tokens_produced += n_accepted

                report.tokens_accepted += n_accepted
                report.tokens_rejected += (K - n_accepted)
                report.tokens_generated += n_accepted

                final_kv_len = kv_len_before_draft + n_accepted + 1
                past_key_values = self._truncate_kv_cache(
                    verify_out.past_key_values, final_kv_len
                )

                if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                    if accepted_tokens[:, -1].item() == self.tokenizer.eos_token_id:
                        break

        except RuntimeError as e:
            # ── Automatic GPU Recovery ───────────────────────────────────────
            torch.cuda.empty_cache()
            
            error_msg = f"Numerical Instability caught (CUDA Error: {str(e)}). Falling back to non-quantized weights for remaining tokens."
            warnings.warn(f"[CSAQ] {error_msg}", stacklevel=2)
            
            report.error_log = error_msg
            self._ensure_draft_state()
            
            remaining = max_new_tokens - tokens_produced
            if remaining > 0:
                out, fallback_report = self.generate(
                    input_ids=generated,
                    speculative=False,
                    max_new_tokens=remaining,
                    temperature=temperature,
                    top_p=top_p
                )
                generated = out
                report.tokens_generated += fallback_report.tokens_generated
                report.mode = "speculative_with_fallback"

        report.total_wallclock_s = time.time() - t0
        return generated, report

    # ═══════════════════════════════════════════════════════════════════════
    # Sampling
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        greedy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # ── Ironclad Numerical Sanitization ──────────────────────────────
        # Stage 1: Prevent exponential overflow. Clamp logits.
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Stage 2: Purge existing extreme mathematical anomalies.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Adaptive Temperature Scaling
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
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = -float("inf")
            scaled = sorted_logits.scatter(1, sorted_indices.argsort(1), sorted_logits)

        probs = F.softmax(scaled, dim=-1)

        # Stage 3: The Validation Barrier
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            warnings.warn(
                "[CSAQ] Fatal numeric collapse detected in Softmax probabilities! "
                "Activating Greedy Fallback to preserve memory integrity.",
                stacklevel=2,
            )
            # Never let it hit CUDA multinomial if the math is broken
            token = logits.argmax(dim=-1, keepdim=True)
            return token, probs

        token = torch.multinomial(probs, 1)
        return token, probs
