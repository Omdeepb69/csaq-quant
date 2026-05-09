"""
csaq/kernels.py — Quantisation primitives with real int8/uint8 bit-packing.

All functions here operate on raw weight tensors.  The critical invariant is:

    dequantize(quantize(W, bits)) ≈ W          (up to rounding error)

Storage contract
────────────────
* bits == 2  → packed into uint8: 4 values per byte, stored as int8 buffer
               with a (scale, zero_point) pair per channel or group.
* bits == 4  → packed into uint8: 2 values per byte, same metadata.
* bits == 8  → stored directly as int8; 1 value per byte.
* bits == 16 → returned as fp16 (bfloat16 on CUDA if available).

The returned ``QuantizedWeight`` dataclass carries everything needed to
re-expand weights at inference time without storing the original fp32 tensor.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# QuantizedWeight — the atomic unit of packed storage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantizedWeight:
    """
    Packed weight buffer + dequantisation metadata.

    Attributes:
        qdata:       Packed uint8 tensor.  Shape depends on packing ratio.
        scales:      Per-channel (or per-group) fp32 scale factors.
                     Shape: ``(out_features,)`` or ``(out_features, n_groups)``.
        zero_points: Integer zero-points matching *scales*.
        bits:        Bit-width this weight was quantised to.
        rows:        Original number of output channels (rows in weight matrix).
        cols:        Original number of input channels (columns).
        group_size:  Column group size (-1 = per-channel).
        shared_scale: True if follower rows share the leader's scale.
    """

    qdata: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    bits: int
    rows: int
    cols: int
    group_size: int = -1
    shared_scale: bool = False

    def dequantize(self) -> torch.Tensor:
        """Reconstruct an fp32 weight matrix from packed storage."""
        return _dequantize(self)

    def element_size_bytes(self) -> float:
        """Average bytes per original weight element (< 1 for sub-byte)."""
        return self.bits / 8.0

    def compression_ratio(self) -> float:
        """Ratio of fp32 size to packed size (higher = more compressed)."""
        return 32.0 / self.bits


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def quantize_per_channel(
    W: torch.Tensor,
    bits: int,
    group_size: int = -1,
) -> QuantizedWeight:
    """
    Asymmetric per-channel (or per-group) quantisation of weight matrix *W*.

    Args:
        W:          fp32 weight tensor of shape ``(out, in)``.
        bits:       Target bit-width.  Must be 2, 4, or 8.
        group_size: If -1 (default) use per-channel scales.  Otherwise split
                    the input dimension into groups of this size.

    Returns:
        :class:`QuantizedWeight` with packed storage ready for dequantisation.
    """
    if bits == 16:
        # fp16 — no packing, just dtype cast stored in uint8 view
        fp16 = W.detach().to(torch.float16)
        dummy_scale = torch.ones(W.shape[0], dtype=torch.float32)
        dummy_zp = torch.zeros(W.shape[0], dtype=torch.int32)
        return QuantizedWeight(
            qdata=fp16.view(torch.uint8),
            scales=dummy_scale,
            zero_points=dummy_zp,
            bits=16,
            rows=W.shape[0],
            cols=W.shape[1],
            group_size=group_size,
        )

    _assert_supported(bits)
    W32 = W.detach().float()
    rows, cols = W32.shape

    if group_size == -1 or group_size >= cols:
        # ── Per-channel ──────────────────────────────────────────────────────
        w_min = W32.min(dim=1).values          # (rows,)
        w_max = W32.max(dim=1).values          # (rows,)
        scales, zero_points = _compute_scale_zp(w_min, w_max, bits)
        Wq = _quantize_rows(W32, scales.unsqueeze(1), zero_points.unsqueeze(1), bits)
        return QuantizedWeight(
            qdata=_pack(Wq, bits),
            scales=scales,
            zero_points=zero_points,
            bits=bits,
            rows=rows,
            cols=cols,
            group_size=-1,
        )
    else:
        # ── Per-group ────────────────────────────────────────────────────────
        n_groups = (cols + group_size - 1) // group_size
        pad = n_groups * group_size - cols
        if pad:
            W32 = torch.nn.functional.pad(W32, (0, pad))
        W_grouped = W32.view(rows, n_groups, group_size)  # (rows, G, gs)

        w_min = W_grouped.min(dim=2).values   # (rows, G)
        w_max = W_grouped.max(dim=2).values
        scales, zero_points = _compute_scale_zp(w_min, w_max, bits)

        Wq_grouped = _quantize_rows(
            W_grouped.reshape(rows * n_groups, group_size),
            scales.reshape(rows * n_groups, 1),
            zero_points.reshape(rows * n_groups, 1),
            bits,
        )
        Wq_grouped = Wq_grouped.view(rows, n_groups * group_size)
        if pad:
            Wq_grouped = Wq_grouped[:, :cols]

        return QuantizedWeight(
            qdata=_pack(Wq_grouped, bits),
            scales=scales,
            zero_points=zero_points,
            bits=bits,
            rows=rows,
            cols=cols,
            group_size=group_size,
        )


def quantize_shared_scale(
    W: torch.Tensor,
    leader_row: torch.Tensor,
    bits: int,
    group_size: int = -1,
) -> QuantizedWeight:
    """
    Quantise follower rows using the *leader* row's scale (shared scale trick).

    This dramatically cuts the per-parameter metadata overhead for cliques,
    because all follower rows reuse one scale/zero-point pair instead of
    storing one per row.

    Args:
        W:          Follower weight rows, shape ``(n_followers, in)``.
        leader_row: The leader row, shape ``(in,)``, used to derive scale.
        bits:       Target bit-width.
        group_size: Per-group override (see :func:`quantize_per_channel`).

    Returns:
        :class:`QuantizedWeight` with ``shared_scale=True``.
    """
    if bits == 16:
        fp16 = W.detach().to(torch.float16)
        dummy_scale = torch.ones(W.shape[0], dtype=torch.float32)
        dummy_zp = torch.zeros(W.shape[0], dtype=torch.int32)
        return QuantizedWeight(
            qdata=fp16.view(torch.uint8),
            scales=dummy_scale,
            zero_points=dummy_zp,
            bits=16,
            rows=W.shape[0],
            cols=W.shape[1],
            group_size=group_size,
            shared_scale=True,
        )

    _assert_supported(bits)
    W32 = W.detach().float()
    L32 = leader_row.detach().float()
    rows, cols = W32.shape

    if group_size == -1 or group_size >= cols:
        l_min = L32.min().unsqueeze(0)    # (1,)
        l_max = L32.max().unsqueeze(0)
        scale, zp = _compute_scale_zp(l_min, l_max, bits)

        # Check for extreme outliers in followers and warn
        follower_range = W32.abs().max().item()
        n_levels = 2**bits - 1
        leader_range = (l_max - l_min).abs().item()
        if leader_range > 0 and follower_range / leader_range > 4.0:
            warnings.warn(
                f"[CSAQ kernels] Follower row outlier: follower/leader range "
                f"ratio={follower_range / leader_range:.1f} (bits={bits}). "
                "Consider increasing group_size or clique_threshold.",
                stacklevel=2,
            )

        Wq = _quantize_rows(W32, scale.unsqueeze(1), zp.unsqueeze(1), bits)
        # Broadcast: all rows share one scale; store it broadcast to (rows,)
        scales_bc = scale.expand(rows).contiguous()
        zp_bc = zp.expand(rows).contiguous()

        return QuantizedWeight(
            qdata=_pack(Wq, bits),
            scales=scales_bc,
            zero_points=zp_bc,
            bits=bits,
            rows=rows,
            cols=cols,
            group_size=-1,
            shared_scale=True,
        )
    else:
        # For grouped shared scale: derive groups from leader only
        n_groups = (cols + group_size - 1) // group_size
        pad = n_groups * group_size - cols

        L_pad = torch.nn.functional.pad(L32.unsqueeze(0), (0, pad)).view(n_groups, group_size)
        W_pad = torch.nn.functional.pad(W32, (0, pad)).view(rows, n_groups, group_size)

        l_min = L_pad.min(dim=1).values   # (G,)
        l_max = L_pad.max(dim=1).values
        scales, zp = _compute_scale_zp(l_min, l_max, bits)  # (G,)

        Wq = _quantize_rows(
            W_pad.reshape(rows * n_groups, group_size),
            scales.unsqueeze(0).expand(rows, -1).reshape(rows * n_groups, 1),
            zp.unsqueeze(0).expand(rows, -1).reshape(rows * n_groups, 1),
            bits,
        ).view(rows, n_groups * group_size)
        if pad:
            Wq = Wq[:, :cols]

        scales_bc = scales.unsqueeze(0).expand(rows, -1).contiguous()
        zp_bc = zp.unsqueeze(0).expand(rows, -1).contiguous()

        return QuantizedWeight(
            qdata=_pack(Wq, bits),
            scales=scales_bc,
            zero_points=zp_bc,
            bits=bits,
            rows=rows,
            cols=cols,
            group_size=group_size,
            shared_scale=True,
        )


def inject_csaq_linear(
    model: nn.Module,
    budget: dict,  # layer_name -> list[clique_dicts with .qweight set]
    verbose: bool = True,
) -> nn.Module:
    """
    Replace ``nn.Linear`` layers that appear in *budget* with
    :class:`CSAQLinear` wrappers.  The original fp32 weights are freed.

    This is the critical step that enables **actual memory savings** —
    the model's parameters now live in packed int8/uint8 buffers.
    """
    replaced = 0
    for name, clique_list in budget.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else _get_submodule(model, parent_name)
        if parent is None:
            continue
        child = getattr(parent, child_name, None)
        if not isinstance(child, nn.Linear):
            continue

        # Build a CSAQLinear from pre-quantised clique data
        new_layer = CSAQLinear.from_cliques(child, clique_list, verbose=verbose)
        setattr(parent, child_name, new_layer)
        replaced += 1

    if verbose:
        print(f"[CSAQ] Replaced {replaced} nn.Linear layers with CSAQLinear.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CSAQLinear — the inference-capable replacement for nn.Linear
# ─────────────────────────────────────────────────────────────────────────────

class CSAQLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` that stores weights in packed
    int8/uint8 format and dequantises on-the-fly during the forward pass.

    Memory layout
    ─────────────
    * ``weight_packed``  — uint8 buffer holding all packed rows.
    * ``weight_scales``  — fp32 scales, one per output row (or per group).
    * ``weight_zp``      — int32 zero-points.
    * High-salience rows can optionally be stored in a fp16 backup buffer for
      the speculative-decoding draft/verify swap (registered as a non-
      persistent buffer so they don't pollute the saved state_dict).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        bias: Optional[torch.Tensor] = None,
        group_size: int = -1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        if bits == 16:
            cols_packed = in_features * 2
        elif bits == 8:
            cols_packed = in_features
        elif bits == 4:
            cols_packed = (in_features + 1) // 2
        elif bits == 2:
            cols_packed = (in_features + 3) // 4
        else:
            cols_packed = 1
            
        n_groups = 1 if group_size == -1 else (in_features + group_size - 1) // group_size

        self.register_buffer("weight_packed", torch.zeros(out_features, cols_packed, dtype=torch.uint8))
        if n_groups == 1:
            self.register_buffer("weight_scales", torch.ones(out_features, dtype=torch.float32))
            self.register_buffer("weight_zp", torch.zeros(out_features, dtype=torch.int32))
        else:
            self.register_buffer("weight_scales", torch.ones(out_features, n_groups, dtype=torch.float32))
            self.register_buffer("weight_zp", torch.zeros(out_features, n_groups, dtype=torch.int32))

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None  # type: ignore[assignment]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for attr in ["weight_packed", "weight_scales", "weight_zp", "_csaq_fp16_backup", "_csaq_quant_stash", "_csaq_hi_rows"]:
            key = f"{prefix}{attr}"
            if key in state_dict:
                current_tensor = getattr(self, attr, None)
                if current_tensor is not None and current_tensor.shape != state_dict[key].shape:
                    current_tensor.resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_cliques(
        cls,
        linear: nn.Linear,
        clique_list: list,
        verbose: bool = False,
    ) -> "CSAQLinear":
        """Build a CSAQLinear by quantising each clique in ``clique_list``."""
        out_f = linear.out_features
        in_f = linear.in_features
        W = linear.weight.data.float()

        # Determine dominant bit (majority vote weighted by element count)
        bit_counts: dict[int, int] = {}
        for c in clique_list:
            b = c.get("bits", 4)
            bit_counts[b] = bit_counts.get(b, 0) + len(c.get("rows", []))
        dominant_bits = max(bit_counts, key=bit_counts.__getitem__)

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bits=dominant_bits,
            bias=linear.bias,
        )

        # Allocate output storage in fp32, then pack at end
        W_reconstructed = torch.zeros_like(W)
        all_scales = torch.ones(out_f, dtype=torch.float32)
        all_zp = torch.zeros(out_f, dtype=torch.int32)

        covered: set[int] = set()
        for c in clique_list:
            rows: list[int] = c["rows"]
            bits: int = c["bits"]
            leader: int = c["leader"]

            if bits == 16:
                W_reconstructed[rows] = W[rows]
                # Scales not meaningful for fp16 passthrough
                continue

            qw = quantize_shared_scale(W[rows], W[leader], bits, layer.group_size)
            W_reconstructed[rows] = qw.dequantize()

            # Store per-row scale/zp (simplified: use leader scale for all in clique)
            sc = qw.scales[:len(rows)]
            zp_ = qw.zero_points[:len(rows)]
            all_scales[rows] = sc
            all_zp[rows] = zp_
            covered.update(rows)

        # Any rows not covered by any clique: quantise at the highest bit option
        uncovered = [i for i in range(out_f) if i not in covered]
        if uncovered:
            qw = quantize_per_channel(W[uncovered], dominant_bits, layer.group_size)
            W_reconstructed[uncovered] = qw.dequantize()
            all_scales[uncovered] = qw.scales[:len(uncovered)]
            all_zp[uncovered] = qw.zero_points[:len(uncovered)]

        # Pack the reconstructed matrix
        Wq_int = _float_to_int(W_reconstructed, all_scales, all_zp, dominant_bits)
        layer.weight_packed = _pack(Wq_int, dominant_bits)
        layer.weight_scales = all_scales
        layer.weight_zp = all_zp
        layer.bits = dominant_bits

        if verbose:
            cr = 32.0 / dominant_bits
            savings = (1.0 - 1.0 / cr) * 100
            print(
                f"  [CSAQLinear] {out_f}×{in_f} → {dominant_bits}-bit "
                f"(compression {cr:.1f}×, memory ↓{savings:.0f}%)"
            )

        del linear.weight  # free original fp32 immediately
        return layer

    def _get_weight_fp32(self) -> torch.Tensor:
        """Unpack stored qdata back to fp32 for the matmul."""
        if self.bits == 16:
            return self.weight_packed.view(torch.float16).float().view(
                self.out_features, self.in_features
            )
        Wq = _unpack(self.weight_packed, self.bits, self.out_features, self.in_features)
        # dequantize: W ≈ (Wq - zp) * scale  (per-channel)
        scale = self.weight_scales.unsqueeze(1).float()
        zp = self.weight_zp.unsqueeze(1).float()
        return (Wq.float() - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight_fp32().to(x.dtype)
        out = nn.functional.linear(x, W, None)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_supported(bits: int) -> None:
    if bits not in (2, 4, 8):
        raise ValueError(
            f"[CSAQ kernels] bits={bits} not supported.  Use 2, 4, 8, or 16."
        )


def _compute_scale_zp(
    w_min: torch.Tensor,
    w_max: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric unsigned quantisation: maps [w_min, w_max] → [0, 2^bits-1]."""
    n_levels = 2**bits - 1
    w_min = w_min.float()
    w_max = w_max.float()
    # Ensure min < max
    w_max = torch.where(w_max > w_min, w_max, w_min + 1e-8)
    scale = (w_max - w_min) / n_levels         # fp32, same shape as w_min
    zp = (-w_min / scale).round().clamp(0, n_levels).to(torch.int32)
    return scale, zp


def _quantize_rows(
    W: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Quantise float tensor W using provided scale & zero_point.
    Returns integer tensor in range [0, 2^bits-1] as int32.
    """
    n_levels = 2**bits - 1
    Wq = ((W / scale) + zp.float()).round().clamp(0, n_levels).to(torch.int32)
    return Wq


def _float_to_int(
    W: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Convert float weight matrix to int using row-wise scale/zp."""
    n_levels = 2**bits - 1
    sc = scales.unsqueeze(1).float()
    zp = zps.unsqueeze(1).float()
    return ((W.float() / sc) + zp).round().clamp(0, n_levels).to(torch.int32)


def _pack(Wq: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack an int32 tensor (values in [0, 2^bits-1]) into a uint8 byte buffer.

    Packing ratios:
        bits=2  → 4 values per byte
        bits=4  → 2 values per byte
        bits=8  → 1 value per byte (cast only)
    """
    Wq = Wq.to(torch.uint8)
    if bits == 8:
        return Wq.contiguous()
    if bits == 4:
        rows, cols = Wq.shape
        if cols % 2:
            Wq = torch.nn.functional.pad(Wq, (0, 1))
        packed = (Wq[:, 0::2] & 0xF) | ((Wq[:, 1::2] & 0xF) << 4)
        return packed.contiguous()
    if bits == 2:
        rows, cols = Wq.shape
        pad = (4 - cols % 4) % 4
        if pad:
            Wq = torch.nn.functional.pad(Wq, (0, pad))
        packed = (
            (Wq[:, 0::4] & 0x3)
            | ((Wq[:, 1::4] & 0x3) << 2)
            | ((Wq[:, 2::4] & 0x3) << 4)
            | ((Wq[:, 3::4] & 0x3) << 6)
        )
        return packed.contiguous()
    raise ValueError(f"Unsupported bits for packing: {bits}")


def _unpack(
    packed: torch.Tensor,
    bits: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Inverse of :func:`_pack`.  Returns int32 tensor of shape (rows, cols)."""
    if bits == 8:
        return packed.view(rows, cols).to(torch.int32)
    if bits == 4:
        lo = (packed & 0xF).to(torch.int32)
        hi = ((packed >> 4) & 0xF).to(torch.int32)
        # Interleave lo/hi columns
        out = torch.zeros(rows, packed.shape[1] * 2, dtype=torch.int32, device=packed.device)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        return out[:, :cols]
    if bits == 2:
        b0 = (packed & 0x3).to(torch.int32)
        b1 = ((packed >> 2) & 0x3).to(torch.int32)
        b2 = ((packed >> 4) & 0x3).to(torch.int32)
        b3 = ((packed >> 6) & 0x3).to(torch.int32)
        out = torch.zeros(rows, packed.shape[1] * 4, dtype=torch.int32, device=packed.device)
        out[:, 0::4] = b0
        out[:, 1::4] = b1
        out[:, 2::4] = b2
        out[:, 3::4] = b3
        return out[:, :cols]
    raise ValueError(f"Unsupported bits for unpacking: {bits}")


def _dequantize(qw: QuantizedWeight) -> torch.Tensor:
    """Reconstruct fp32 weight from a QuantizedWeight."""
    if qw.bits == 16:
        return qw.qdata.view(torch.float16).float().view(qw.rows, qw.cols)
    Wq = _unpack(qw.qdata, qw.bits, qw.rows, qw.cols).float()
    if qw.group_size == -1 or qw.scales.dim() == 1:
        scale = qw.scales.float().unsqueeze(1)
        zp = qw.zero_points.float().unsqueeze(1)
        return (Wq - zp) * scale
    else:
        # Per-group: scales shape (rows, n_groups)
        n_groups = qw.scales.shape[1]
        gs = qw.cols // n_groups
        Wq_g = Wq.view(qw.rows, n_groups, gs)
        sc = qw.scales.float().unsqueeze(2)
        zp_ = qw.zero_points.float().unsqueeze(2)
        return ((Wq_g - zp_) * sc).view(qw.rows, qw.cols)


def _get_submodule(model: nn.Module, name: str) -> Optional[nn.Module]:
    mod: nn.Module = model
    for part in name.split("."):
        child = getattr(mod, part, None)
        if child is None:
            return None
        mod = child
    return mod
