"""
csaq/config.py — CSAQConfig: typed, validated, HuggingFace-compatible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import PretrainedConfig


# Valid bit widths that have real storage representations in PyTorch / packed int8.
# 1-bit is sign-only (stored in int8), 2-bit is packed 4-per-byte.
_SUPPORTED_BITS: frozenset[int] = frozenset({2, 4, 8, 16})


class CSAQConfig(PretrainedConfig):
    """
    Configuration for Causal Salience-Aware Quantization.

    Registered as a HuggingFace ``quantization_config`` so that quantized
    models can be reloaded with ``AutoModelForCausalLM.from_pretrained``.

    Args:
        target_bits: Target *average* bits-per-weight across the full model.
            The solver will distribute higher precision to salient weight
            cliques and lower precision to follower weights to hit this budget.
        bit_options: Discrete bit-widths the solver may assign.
            **Only ``{2, 4, 8, 16}`` are supported** — these map to real
            PyTorch int8/uint8 packed storage.  1-bit is intentionally
            removed: sign-only quantisation on LLMs causes catastrophic
            accuracy loss and has no matching runtime kernel.
        clique_threshold: Jaccard similarity threshold ``[0, 1]`` above which
            two output channels are considered co-activated and grouped into
            the same clique.  Higher = fewer, larger cliques.
        auto_scale_memory: If ``True`` (default), CSAQ scales calibration
            batch count based on available CPU RAM to avoid OOM.
        speculative_lookahead: Number of draft tokens generated per
            speculative decoding block.
        speculative_temperature: Softmax temperature used in the draft and
            verification passes (0 = greedy).
        salience_alpha: Scaling factor for the activation-sparsity mask
            fraction used during profiling.  1.0 = top-10 % of activations.
        protection_floor: Fraction of the most salient rows in each layer
            that are *always* protected at ≥ 8-bit, regardless of the budget.
        group_size: Number of consecutive weight columns that share a scale
            factor.  -1 = per-channel (no grouping, default).  128 or 64 are
            common GPTQ-style group sizes for better accuracy at low bits.
    """

    model_type: str = "csaq"
    quant_type: str = "csaq"  # for HF quantization_config detection

    def __init__(
        self,
        target_bits: float = 4.0,
        bit_options: Optional[List[int]] = None,
        clique_threshold: float = 0.85,
        auto_scale_memory: bool = True,
        speculative_lookahead: int = 4,
        speculative_temperature: float = 1.0,
        salience_alpha: float = 1.0,
        protection_floor: float = 0.10,
        group_size: int = -1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)

        # ── Bit options ─────────────────────────────────────────────────────
        if bit_options is None:
            bit_options = [4, 8, 16]
        unsupported = sorted(set(bit_options) - _SUPPORTED_BITS)
        if unsupported:
            warnings.warn(
                f"[CSAQConfig] bit_options contains unsupported widths "
                f"{unsupported} — they will be silently ignored.  "
                f"Supported: {sorted(_SUPPORTED_BITS)}.",
                stacklevel=2,
            )
            bit_options = [b for b in bit_options if b in _SUPPORTED_BITS]
        if not bit_options:
            raise ValueError(
                "[CSAQConfig] No valid bit_options remain after filtering.  "
                f"Choose from {sorted(_SUPPORTED_BITS)}."
            )
        bit_options = sorted(set(bit_options))

        # ── target_bits validation ───────────────────────────────────────────
        min_b, max_b = float(min(bit_options)), float(max(bit_options))
        if not (min_b <= target_bits <= max_b):
            raise ValueError(
                f"[CSAQConfig] target_bits={target_bits} is outside the range "
                f"[{min_b}, {max_b}] implied by bit_options={bit_options}."
            )

        # ── Threshold / fraction validation ──────────────────────────────────
        if not (0.0 < clique_threshold <= 1.0):
            raise ValueError("[CSAQConfig] clique_threshold must be in (0, 1].")
        if not (0.0 <= protection_floor < 1.0):
            raise ValueError("[CSAQConfig] protection_floor must be in [0, 1).")

        self.target_bits = target_bits
        self.bit_options = bit_options
        self.clique_threshold = clique_threshold
        self.auto_scale_memory = auto_scale_memory
        self.speculative_lookahead = speculative_lookahead
        self.speculative_temperature = speculative_temperature
        self.salience_alpha = salience_alpha
        self.protection_floor = protection_floor
        self.group_size = group_size

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def min_bits(self) -> int:
        return min(self.bit_options)

    @property
    def max_bits(self) -> int:
        return max(self.bit_options)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CSAQConfig(target_bits={self.target_bits}, "
            f"bit_options={self.bit_options}, "
            f"clique_threshold={self.clique_threshold}, "
            f"group_size={self.group_size})"
        )
