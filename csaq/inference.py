"""
csaq/inference.py — Self-Speculative Decoding Engine (v0.3.9)
Hardened version with full state restoration and numerical safety guards.
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
    """

    def __init__(
        self,
        model: nn.Module,
        causal_map: Dict[str, List[int]],
        tokenizer=None,
        verbose: bool = True
    ):
        self.model = model
        self.causal_map = causal_map
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = next(model.parameters()).device
        self._in_verify = False
        
        # Bug 1 Fix: Correct state initialization
        self.telemetry = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "backpressure_events": 0, 
        }
        self._warmup_complete = False
        self._hooks: List[_HookEntry] = []

        # Bug 2 Fix: Move hook building BEFORE warmup
        self._build_hooks()
        
        if not self._hooks:
            warnings.warn(
                "[CSAQ] CSAQInferenceEngine initialized with 0 hooked layers.",
                stacklevel=2,
            )

    def _build_hooks(self):
        """Pre-calculate hooked layers and valid hi-salience rows."""
        self._hooks.clear()
        for module_name in self.causal_map:
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

    def warmup(self, n: int = 5):
        """Warmup the CUDA kernels and Python loops to stabilize metrics."""
        if self._warmup_complete or not self._hooks:
            return
        # Use a real token ID from tokenizer if possible
        dummy_in = torch.tensor([[1]], device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self.generate(dummy_in, max_new_tokens=4, speculative=True)
        self._warmup_complete = True
        if self.verbose:
            print("[CSAQ] Inference Engine Warmup Complete.")

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
            truncated.append(tuple(t[:, :, :accepted_len, :] for t in layer_past))
        return tuple(truncated)

    @staticmethod
    def _get_kv_len(past_key_values) -> int:
        if past_key_values is None: return 0
        if hasattr(past_key_values, "get_seq_length"): return past_key_values.get_seq_length()
        if hasattr(past_key_values, "key_cache"):
            if len(past_key_values.key_cache) == 0: return 0
            return past_key_values.key_cache[0].shape[2]
        return past_key_values[0][0].shape[2]

    # ═══════════════════════════════════════════════════════════════════════
    # .generate()
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
        gen_kwargs = dict(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), **kwargs)
        if temperature > 0: gen_kwargs["temperature"] = temperature
        if top_p < 1.0: gen_kwargs["top_p"] = top_p
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
            gen_kwargs.setdefault("pad_token_id", self.model.config.eos_token_id)

        output = self.model.generate(**gen_kwargs)
        elapsed = time.time() - t0
        p_len = input_ids.shape[-1] if input_ids is not None else 0
        n_new = max(0, output.shape[-1] - p_len)
        return output, SpeculativeReport(tokens_generated=n_new, tokens_accepted=n_new, total_wallclock_s=elapsed, mode="standard")

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        lookahead: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        if not self._hooks:
            return self.generate(input_ids=input_ids, speculative=False, max_new_tokens=max_new_tokens)

        report = SpeculativeReport()
        t0 = time.time()
        self.model.eval()
        self._ensure_draft_state()
        generated = input_ids.clone().to(self.device)
        past_key_values = None
        greedy = (temperature == 0.0)
        tokens_produced = 0

        try:
            while tokens_produced < max_new_tokens:
                K = min(lookahead, max_new_tokens - tokens_produced)
                if K <= 0: break

                draft_tokens, draft_logits_list = [], []
                draft_input = generated[:, -1:]
                old_kv_len = self._get_kv_len(past_key_values)

                # DRAFT
                for _ in range(K):
                    out = self.model(input_ids=draft_input, past_key_values=past_key_values, use_cache=True)
                    if out.logits.is_cuda: torch.cuda.synchronize()
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    report.draft_calls += 1
                    token, _ = self._sample(logits, temperature, top_p, greedy)
                    draft_tokens.append(token)
                    draft_logits_list.append(logits)
                    draft_input = token if token.dim() == 2 else token.unsqueeze(0)

                draft_token_ids = torch.cat(draft_tokens, dim=-1)

                # VERIFY
                verify_kv = self._truncate_kv_cache(past_key_values, old_kv_len)
                verify_input = torch.cat([generated[:, -(K + 1):-K], draft_token_ids], dim=-1)

                try:
                    self._swap_to_verify()
                    v_out = self.model(input_ids=verify_input, past_key_values=verify_kv, use_cache=True)
                    if v_out.logits.is_cuda: torch.cuda.synchronize()
                    v_logits = v_out.logits
                    report.verify_calls += 1
                finally:
                    self._swap_to_draft()

                # REJECTION SAMPLING
                n_accepted = 0
                for k in range(K):
                    d_logit_k, v_logit_k, d_token_k = draft_logits_list[k], v_logits[:, k + 1, :], draft_token_ids[:, k]
                    if greedy:
                        v_token = v_logit_k.argmax(dim=-1)
                        if v_token.item() == d_token_k.item(): n_accepted += 1
                        else:
                            draft_token_ids[:, k] = v_token
                            n_accepted += 1
                            break
                    else:
                        p_draft, p_verify = F.softmax(d_logit_k / temperature, dim=-1), F.softmax(v_logit_k / temperature, dim=-1)
                        token_idx = d_token_k.item()
                        ratio = p_verify[0, token_idx] / p_draft[0, token_idx].clamp(min=1e-10)
                        if torch.rand(1, device=self.device).item() < min(1.0, ratio.item()):
                            n_accepted += 1
                        else:
                            res = (p_verify - p_draft).clamp(min=0.0)
                            s = res.sum()
                            if s > 1e-8: draft_token_ids[:, k] = torch.multinomial(res / s, 1).squeeze(-1)
                            else: draft_token_ids[:, k] = v_logit_k.argmax(dim=-1)
                            n_accepted += 1
                            break

                # COMMIT
                accepted_tokens = draft_token_ids[:, :n_accepted]
                generated = torch.cat([generated, accepted_tokens], dim=-1)
                tokens_produced += n_accepted
                report.tokens_accepted += n_accepted
                report.tokens_rejected += (K - n_accepted)
                report.tokens_generated += n_accepted

                rate = n_accepted / max(K, 1)
                if rate < 0.20:
                    self.telemetry["backpressure_events"] += 1
                    if self.verbose and tokens_produced % 50 == 0:
                        print(f"  [CSAQ Telemetry] ⚠ Backpressure (Acceptance: {rate*100:.1f}%)")

                past_key_values = self._truncate_kv_cache(v_out.past_key_values, old_kv_len + n_accepted + 1)
                
                # Bug 7 Fix: Guard against n_accepted == 0 for EOS check
                if n_accepted > 0 and self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                    if accepted_tokens[:, -1].item() == self.tokenizer.eos_token_id:
                        break

        except RuntimeError as e:
            # Bug 9 Fix: Guard empty_cache
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            error_msg = f"Numerical Instability caught (CUDA Error: {str(e)}). Falling back."
            warnings.warn(f"[CSAQ] {error_msg}")
            report.error_log = error_msg
            self._ensure_draft_state()
            rem = max(0, max_new_tokens - tokens_produced)
            if rem > 0:
                gen, f_rep = self.generate(input_ids=generated, speculative=False, max_new_tokens=rem)
                generated = gen
                report.tokens_generated += f_rep.tokens_generated
        
        report.total_wallclock_s = time.time() - t0
        return generated, report

    @staticmethod
    def _sample(logits: torch.Tensor, temp: float, top_p: float, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        if greedy: return logits.argmax(dim=-1, keepdim=True), F.softmax(logits, dim=-1)
        scaled = logits / max(temp, 1e-8)
        if top_p < 1.0:
            s_logits, s_indices = torch.sort(scaled, descending=True)
            cumulative = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            s_logits[cumulative - F.softmax(s_logits, dim=-1) > top_p] = -float("inf")
            scaled = s_logits.scatter(1, s_indices.argsort(1), s_logits)
        probs = F.softmax(scaled, dim=-1)
        if torch.isnan(probs).any() or (probs < 0).any():
            return logits.argmax(dim=-1, keepdim=True), probs
        return torch.multinomial(probs, 1), probs
