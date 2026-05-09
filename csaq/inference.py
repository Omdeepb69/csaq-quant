"""
csaq/inference.py — Self-Speculative Decoding engine.

The key idea: a CSAQ model already contains two "personalities" in one:
  - Draft mode:  quantised (lower quality, faster) — used for lookahead tokens.
  - Verify mode: high-salience rows are restored to fp16 (higher quality).

By swapping only the high-salience row buffers (rather than maintaining a
separate draft model), we gain speculative speedup with near-zero memory
overhead beyond the quantised model itself.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import CSAQLinear

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# SpeculativeReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeculativeReport:
    """
    Tracks real performance metrics for a speculative decoding session.

    All latency figures are wall-clock seconds measured from the first
    ``generate`` call to completion.
    """

    tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    draft_calls: int = 0
    verify_calls: int = 0
    total_wallclock_s: float = 0.0
    mode: str = "speculative"
    error_log: str = ""
    _token_times: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the verifier."""
        total = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / max(total, 1)

    @property
    def speedup_factor(self) -> float:
        """Average tokens generated per forward pass (draft + verify)."""
        return self.tokens_generated / max(self.draft_calls + self.verify_calls, 1)

    @property
    def inter_token_latency_ms(self) -> float:
        """Average milliseconds per generated token."""
        return (self.total_wallclock_s * 1e3) / max(self.tokens_generated, 1)

    @property
    def block_efficiency(self) -> float:
        """Average accepted tokens per verify call."""
        return self.tokens_accepted / max(self.verify_calls, 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self._token_times:
            return float("nan")
        return float(np.percentile(self._token_times, 95)) * 1000.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_wallclock_s <= 0:
            return 0.0
        return self.tokens_generated / self.total_wallclock_s

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tokens_generated": self.tokens_generated,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_factor": round(self.speedup_factor, 2),
            "inter_token_latency_ms": round(self.inter_token_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2) if not np.isnan(self.p95_latency_ms) else "nan",
            "tokens_per_second": round(self.tokens_per_second, 2),
            "block_efficiency": round(self.block_efficiency, 2),
            "draft_calls": self.draft_calls,
            "verify_calls": self.verify_calls,
            "wallclock_s": round(self.total_wallclock_s, 3),
            "error_log": self.error_log,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpeculativeReport(tokens={self.tokens_generated}, "
            f"acceptance={self.acceptance_rate:.2%}, "
            f"speedup={self.speedup_factor:.2f}x, "
            f"latency={self.inter_token_latency_ms:.1f}ms/tok)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hook entry type: (CSAQLinear, hi_rows, fp16_backup, quant_stash)
# ─────────────────────────────────────────────────────────────────────────────

_HookEntry = Tuple[CSAQLinear, torch.Tensor, torch.Tensor, torch.Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# CSAQInferenceEngine
# ─────────────────────────────────────────────────────────────────────────────

class CSAQInferenceEngine:
    """
    Self-Speculative Decoding engine for CSAQ-quantized models.

    Usage::

        engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer)
        output, report = engine.generate(input_ids, max_new_tokens=200,
                                          speculative=True, lookahead=4)
        print(report.summary())

    The engine works by exploiting the dual-weight structure created by
    :func:`~csaq.core.apply_csaq`:

    * **Draft pass** — the model runs with fully-quantised weights.
      Cheap, fast, but slightly less accurate.
    * **Verify pass** — the engine swaps in fp16 backups for the top-p%
      most salient weight rows, making those layers near-lossless.
      Only the high-salience buffers are touched, not the full parameter set.

    If the hooked layers are fewer than expected (e.g., the model was not
    quantised with CSAQ), the engine falls back gracefully to standard
    autoregressive generation.
    """

    def __init__(
        self,
        model: nn.Module,
        causal_map: Dict[str, List[int]],
        tokenizer: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.causal_map = causal_map
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = next(model.parameters()).device
        self._in_verify: bool = False
        self._warmup_complete: bool = False

        self.telemetry: Dict[str, int] = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "backpressure_events": 0,
        }

        self._hooks: List[_HookEntry] = []
        self._build_hooks()

        if not self._hooks:
            warnings.warn(
                "[CSAQ] CSAQInferenceEngine found 0 hooked layers.  "
                "Speculative decoding will fall back to standard generation.  "
                "Make sure the model was quantised with csaq.quantize().",
                stacklevel=2,
            )

    # ── Hook discovery ────────────────────────────────────────────────────────

    def _build_hooks(self) -> None:
        self._hooks.clear()
        for module_name in self.causal_map:
            module = self._resolve_module(module_name)
            if module is None:
                continue
            if not isinstance(module, CSAQLinear):
                continue
            if not (
                hasattr(module, "_csaq_fp16_backup")
                and hasattr(module, "_csaq_quant_stash")
                and hasattr(module, "_csaq_hi_rows")
            ):
                continue
            self._hooks.append((
                module,
                module._csaq_hi_rows,
                module._csaq_fp16_backup,
                module._csaq_quant_stash,
            ))

    def _resolve_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        if parts and parts[-1] == "weight":
            parts = parts[:-1]
        mod: nn.Module = self.model
        for p in parts:
            child = getattr(mod, p, None)
            if child is None:
                return None
            mod = child
        return mod if isinstance(mod, nn.Module) else None

    # ── Weight swapping ───────────────────────────────────────────────────────

    def _swap_to_verify(self) -> None:
        """Restore fp16 backups for high-salience rows (verify pass)."""
        for module, rows, fp16_backup, _ in self._hooks:
            W = module._get_weight_fp32()
            W[rows] = fp16_backup.float()
            # Write back (CSAQLinear keeps its dequantised reconstruction in memory)
            module._verify_weight_override = W
        self._in_verify = True

    def _swap_to_draft(self) -> None:
        """Clear any verify-pass overrides (return to quantised draft)."""
        for module, _, _, _ in self._hooks:
            if hasattr(module, "_verify_weight_override"):
                del module._verify_weight_override
        self._in_verify = False

    def _ensure_draft_state(self) -> None:
        if self._in_verify:
            self._swap_to_draft()

    # ── KV-cache utilities ────────────────────────────────────────────────────

    @staticmethod
    def _kv_len(past_key_values: Optional[Any]) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if hasattr(past_key_values, "key_cache"):
            kc = past_key_values.key_cache
            return kc[0].shape[2] if kc else 0
        return past_key_values[0][0].shape[2]

    @staticmethod
    def _truncate_kv(past_key_values: Optional[Any], length: int) -> Optional[Any]:
        if past_key_values is None:
            return None
        if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
            if hasattr(past_key_values, "crop"):
                past_key_values.crop(length)
                return past_key_values
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :length, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :length, :]
            return past_key_values
        # Tuple-of-tuples format
        return tuple(tuple(t[:, :, :length, :] for t in layer) for layer in past_key_values)

    # ── Sampling ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        greedy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Safe sampling from logits with numerical stability guards.

        Returns (token_id, probabilities) where token_id has shape (1, 1).
        """
        logits = torch.clamp(logits, -50.0, 50.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        if greedy:
            token = logits.argmax(dim=-1, keepdim=True)
            return token, F.softmax(logits, dim=-1)

        scaled = logits / max(temperature, 1e-8)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens where cumulative prob exceeds top_p
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            scaled = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(scaled, dim=-1)

        # Numerical fallback
        if torch.isnan(probs).any() or (probs < 0).any() or probs.sum() < 1e-8:
            return logits.argmax(dim=-1, keepdim=True), probs

        return torch.multinomial(probs, 1), probs

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self, n: int = 3) -> None:
        """
        Run short dummy generations to warm up CUDA kernels.
        Call this before timing-sensitive benchmarks.
        """
        if self._warmup_complete:
            return
        dummy = torch.tensor([[1]], device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self.generate(dummy, max_new_tokens=2, speculative=bool(self._hooks))
        self._warmup_complete = True
        if self.verbose:
            print("[CSAQ] Inference engine warmup complete.")

    # ── Standard generation ───────────────────────────────────────────────────

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        speculative: bool = False,
        lookahead: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        """
        Generate tokens.  When ``speculative=True`` and the model has CSAQ
        hooks installed, uses self-speculative decoding for a speedup.

        Args:
            input_ids:      Prompt token ids, shape ``(1, seq_len)``.
            speculative:    Use speculative decoding (requires CSAQ hooks).
            lookahead:      Number of draft tokens per speculative block.
            temperature:    Sampling temperature (0 = greedy).
            top_p:          Nucleus sampling threshold.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs:       Passed to ``model.generate()`` in standard mode.

        Returns:
            ``(generated_ids, SpeculativeReport)``
        """
        if speculative and self._hooks:
            return self._generate_speculative(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                lookahead=lookahead,
                temperature=temperature,
                top_p=top_p,
            )

        # Standard autoregressive generation
        self._ensure_draft_state()
        t0 = time.time()

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            **kwargs,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p

        pad_id = (
            (self.tokenizer.pad_token_id if self.tokenizer else None)
            or getattr(getattr(self.model, "config", None), "eos_token_id", None)
        )
        if pad_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_id)

        output: torch.Tensor = self.model.generate(**gen_kwargs)
        elapsed = time.time() - t0
        p_len = input_ids.shape[-1] if input_ids is not None else 0
        n_new = max(0, output.shape[-1] - p_len)
        return output, SpeculativeReport(
            tokens_generated=n_new,
            tokens_accepted=n_new,
            total_wallclock_s=elapsed,
            mode="standard",
        )

    # ── Speculative generation ────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        lookahead: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[torch.Tensor, SpeculativeReport]:
        """
        Self-speculative decoding loop.

        Each iteration:
        1. Draft:  generate K tokens with quantised weights.
        2. Verify: run full verify pass (fp16 hi-salience rows) on all K+1
                   positions in one batched forward pass.
        3. Accept/reject each draft token via token-level rejection sampling.
        4. Commit accepted tokens and truncate the KV cache accordingly.
        """
        report = SpeculativeReport()
        t0 = time.time()
        greedy = temperature == 0.0

        self.model.eval()
        self._ensure_draft_state()

        generated = input_ids.clone().to(self.device)
        past_key_values: Optional[Any] = None
        tokens_produced = 0
        last_time = time.time()

        eos_id: Optional[int] = (
            self.tokenizer.eos_token_id if self.tokenizer else None
        ) or getattr(getattr(self.model, "config", None), "eos_token_id", None)

        try:
            while tokens_produced < max_new_tokens:
                K = min(lookahead, max_new_tokens - tokens_produced)
                if K <= 0:
                    break

                # ── DRAFT ────────────────────────────────────────────────────
                draft_tokens: List[torch.Tensor] = []
                draft_logits: List[torch.Tensor] = []
                draft_input = generated[:, -1:]
                kv_len_before = self._kv_len(past_key_values)

                for _ in range(K):
                    out = self.model(
                        input_ids=draft_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    if out.logits.is_cuda:
                        torch.cuda.synchronize()
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    report.draft_calls += 1
                    token, _ = self._sample(logits, temperature, top_p, greedy)
                    draft_tokens.append(token)
                    draft_logits.append(logits)
                    draft_input = token if token.dim() == 2 else token.unsqueeze(0)

                draft_ids = torch.cat(draft_tokens, dim=-1)  # (1, K)

                # ── VERIFY ───────────────────────────────────────────────────
                verify_kv = self._truncate_kv(past_key_values, kv_len_before)
                # Feed [last context token] + [K draft tokens] together
                verify_input = torch.cat(
                    [generated[:, -1:], draft_ids], dim=-1
                )  # (1, K+1)

                try:
                    self._swap_to_verify()
                    v_out = self.model(
                        input_ids=verify_input,
                        past_key_values=verify_kv,
                        use_cache=True,
                    )
                    if v_out.logits.is_cuda:
                        torch.cuda.synchronize()
                    v_logits = v_out.logits   # (1, K+1, vocab)
                    report.verify_calls += 1
                finally:
                    self._swap_to_draft()

                # ── REJECTION SAMPLING ────────────────────────────────────────
                n_accepted = 0
                for k in range(K):
                    d_logit = draft_logits[k]           # (1, vocab)
                    v_logit = v_logits[:, k + 1, :]     # (1, vocab)  — verifier's prediction at position k
                    d_token = draft_ids[:, k]            # (1,)

                    if greedy:
                        # Greedy: accept if verifier agrees, else substitute
                        v_token = v_logit.argmax(dim=-1)
                        if v_token.item() == d_token.item():
                            n_accepted += 1
                        else:
                            draft_ids[:, k] = v_token
                            n_accepted += 1
                            break
                    else:
                        # Stochastic rejection sampling
                        p_draft = F.softmax(d_logit / max(temperature, 1e-8), dim=-1)
                        p_verify = F.softmax(v_logit / max(temperature, 1e-8), dim=-1)
                        token_idx = d_token.item()
                        ratio = (
                            p_verify[0, token_idx] / p_draft[0, token_idx].clamp(min=1e-10)
                        )
                        u = torch.rand(1, device=self.device).item()
                        if u < min(1.0, ratio.item()):
                            n_accepted += 1
                        else:
                            # Sample from residual distribution
                            residual = (p_verify - p_draft).clamp(min=0.0)
                            s = residual.sum()
                            if s > 1e-8:
                                draft_ids[:, k] = torch.multinomial(residual / s, 1).squeeze(-1)
                            else:
                                draft_ids[:, k] = v_logit.argmax(dim=-1)
                            n_accepted += 1
                            break

                # ── COMMIT ────────────────────────────────────────────────────
                accepted = draft_ids[:, :n_accepted]
                generated = torch.cat([generated, accepted], dim=-1)
                tokens_produced += n_accepted
                report.tokens_accepted += n_accepted
                report.tokens_rejected += K - n_accepted
                report.tokens_generated += n_accepted
                
                now = time.time()
                dt = now - last_time
                if n_accepted > 0:
                    report._token_times.extend([dt / n_accepted] * n_accepted)
                last_time = now

                self.telemetry["total_tokens"] += n_accepted
                self.telemetry["accepted_tokens"] += n_accepted

                rate = n_accepted / max(K, 1)
                if rate < 0.20:
                    self.telemetry["backpressure_events"] += 1
                    if self.verbose and tokens_produced % 64 == 0:
                        print(
                            f"  [CSAQ] Low acceptance ({rate*100:.1f}%) — "
                            "consider reducing lookahead or clique_threshold."
                        )

                past_key_values = self._truncate_kv(
                    v_out.past_key_values,
                    kv_len_before + n_accepted + 1,
                )

                # EOS check
                if n_accepted > 0 and eos_id is not None:
                    if accepted[:, -1].item() == eos_id:
                        break

        except RuntimeError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msg = f"RuntimeError during speculative decoding: {exc}"
            warnings.warn(f"[CSAQ] {msg}  Falling back to standard generation.")
            report.error_log = msg
            self._ensure_draft_state()
            rem = max(0, max_new_tokens - tokens_produced)
            if rem > 0:
                fallback, fb_report = self.generate(
                    input_ids=generated,
                    speculative=False,
                    max_new_tokens=rem,
                )
                generated = fallback
                report.tokens_generated += fb_report.tokens_generated

        report.total_wallclock_s = time.time() - t0
        return generated, report
