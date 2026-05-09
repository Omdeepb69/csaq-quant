"""
tests/test_kernels.py — Unit tests for quantisation kernels.

Tests the critical invariant:  dequantize(quantize(W)) ≈ W
across all supported bit widths and packing modes.
"""

from __future__ import annotations

import pytest
import torch

from csaq.kernels import (
    QuantizedWeight,
    _pack,
    _unpack,
    quantize_per_channel,
    quantize_shared_scale,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_weight(rows: int = 32, cols: int = 64, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, cols)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Pack / unpack round-trip
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip(bits: int) -> None:
    """Packing then unpacking must reproduce the original integer tensor."""
    rows, cols = 16, 64
    n_levels = 2**bits - 1
    torch.manual_seed(0)
    Wq = torch.randint(0, n_levels + 1, (rows, cols), dtype=torch.int32)

    packed = _pack(Wq, bits)
    restored = _unpack(packed, bits, rows, cols)

    assert restored.shape == (rows, cols), "Shape mismatch after unpack"
    assert torch.all(restored == Wq), (
        f"Pack→Unpack not lossless for bits={bits}. "
        f"Max diff: {(restored - Wq).abs().max().item()}"
    )


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_odd_columns(bits: int) -> None:
    """Odd column counts must be handled via padding without corrupting data."""
    rows, cols = 8, 37   # deliberately odd
    n_levels = 2**bits - 1
    torch.manual_seed(1)
    Wq = torch.randint(0, n_levels + 1, (rows, cols), dtype=torch.int32)

    packed = _pack(Wq, bits)
    restored = _unpack(packed, bits, rows, cols)

    assert restored.shape == (rows, cols)
    assert torch.all(restored == Wq), (
        f"Odd-column pack→unpack failed for bits={bits}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# quantize_per_channel round-trip
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_per_channel_roundtrip_mse(bits: int) -> None:
    """
    Quantisation MSE must be below a reasonable threshold.
    8-bit should be extremely close; 4-bit should not distort by more than ~5%.
    """
    W = _random_weight(32, 128)
    qw = quantize_per_channel(W, bits)
    W_hat = qw.dequantize()

    assert W_hat.shape == W.shape, "Shape mismatch after dequantise"
    mse = _mse(W, W_hat)
    threshold = 0.001 if bits == 8 else 0.05
    assert mse < threshold, (
        f"MSE={mse:.6f} exceeds threshold={threshold} for bits={bits}"
    )


def test_quantize_per_channel_16bit_passthrough() -> None:
    """bits=16 must return a tensor numerically equal to the input (fp16 cast)."""
    W = _random_weight(8, 32)
    qw = quantize_per_channel(W, 16)
    W_hat = qw.dequantize()
    # fp16 cast and back introduces minor error
    assert _mse(W, W_hat) < 1e-4


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_per_channel_shape_preserved(bits: int) -> None:
    rows, cols = 24, 96
    W = _random_weight(rows, cols)
    qw = quantize_per_channel(W, bits)
    assert qw.dequantize().shape == (rows, cols)


@pytest.mark.parametrize("bits,group_size", [(4, 32), (8, 64)])
def test_quantize_per_group(bits: int, group_size: int) -> None:
    """Per-group quantisation should have lower MSE than per-channel at same bits."""
    W = _random_weight(16, 128)

    qw_chan = quantize_per_channel(W, bits, group_size=-1)
    qw_grp  = quantize_per_channel(W, bits, group_size=group_size)

    mse_chan = _mse(W, qw_chan.dequantize())
    mse_grp  = _mse(W, qw_grp.dequantize())

    assert qw_grp.dequantize().shape == W.shape
    # Per-group should always be ≤ per-channel MSE (more granular scaling)
    assert mse_grp <= mse_chan + 1e-6, (
        f"Per-group MSE ({mse_grp:.6f}) > per-channel MSE ({mse_chan:.6f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# quantize_shared_scale
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [4, 8])
def test_shared_scale_roundtrip(bits: int) -> None:
    """Shared-scale quantisation must produce valid dequantised output."""
    torch.manual_seed(7)
    W = torch.randn(16, 64)
    leader = W[0]

    qw = quantize_shared_scale(W, leader, bits)
    W_hat = qw.dequantize()

    assert W_hat.shape == W.shape
    assert qw.shared_scale is True
    mse = _mse(W, W_hat)
    # Shared-scale quantisation intentionally accepts slightly more error than
    # per-channel because all rows share one scale derived from the leader row.
    # The trade-off is reduced metadata overhead for clique members.
    threshold = 0.01 if bits == 8 else 0.15
    assert mse < threshold, f"Shared-scale MSE={mse:.6f} > threshold={threshold}"


def test_shared_scale_outlier_warning() -> None:
    """Extreme outlier followers should trigger a UserWarning."""
    torch.manual_seed(99)
    leader = torch.randn(64) * 0.01        # very small scale
    W = torch.randn(8, 64) * 10.0          # followers are 1000× bigger

    with pytest.warns(UserWarning, match="Follower row outlier"):
        quantize_shared_scale(W, leader, bits=4)


# ─────────────────────────────────────────────────────────────────────────────
# QuantizedWeight dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_quantized_weight_compression_ratio() -> None:
    W = _random_weight(8, 32)
    qw = quantize_per_channel(W, 4)
    assert abs(qw.compression_ratio() - 8.0) < 1e-6  # 32/4 = 8×

    qw8 = quantize_per_channel(W, 8)
    assert abs(qw8.compression_ratio() - 4.0) < 1e-6  # 32/8 = 4×


def test_quantized_weight_element_size() -> None:
    W = _random_weight(8, 32)
    qw = quantize_per_channel(W, 4)
    assert abs(qw.element_size_bytes() - 0.5) < 1e-9   # 4/8 = 0.5 bytes


# ─────────────────────────────────────────────────────────────────────────────
# CSAQLinear
# ─────────────────────────────────────────────────────────────────────────────

def test_csaq_linear_forward() -> None:
    """CSAQLinear forward must produce output of correct shape."""
    import torch.nn as nn
    from csaq.kernels import CSAQLinear

    in_f, out_f = 64, 32
    linear = nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(3)
    linear.weight.data = torch.randn(out_f, in_f)

    # Build a minimal clique covering all rows
    clique_list = [{
        "rows": list(range(out_f)),
        "bits": 4,
        "leader": 0,
    }]

    csaq_lin = CSAQLinear.from_cliques(linear, clique_list, verbose=False)

    x = torch.randn(2, 10, in_f)
    out = csaq_lin(x)
    assert out.shape == (2, 10, out_f), f"Unexpected output shape: {out.shape}"


def test_csaq_linear_output_numerics() -> None:
    """CSAQLinear at 8-bit should be very close to fp32 linear."""
    import torch.nn as nn
    from csaq.kernels import CSAQLinear

    torch.manual_seed(42)
    in_f, out_f = 128, 64
    linear = nn.Linear(in_f, out_f, bias=False)
    linear.weight.data = torch.randn(out_f, in_f) * 0.02

    W_fp32 = linear.weight.data.clone()
    clique_list = [{"rows": list(range(out_f)), "bits": 8, "leader": 0}]
    csaq_lin = CSAQLinear.from_cliques(linear, clique_list, verbose=False)

    x = torch.randn(1, 16, in_f)
    with torch.no_grad():
        out_q = csaq_lin(x)
        out_fp = torch.nn.functional.linear(x, W_fp32)

    mse = _mse(out_q, out_fp)
    # 8-bit with shared scale introduces small but non-zero rounding versus
    # per-channel per-element fp32.  5e-4 is a conservative but correct bound.
    assert mse < 5e-4, f"8-bit linear output MSE too high: {mse:.6f}"
