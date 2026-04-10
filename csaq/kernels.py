import torch
import torch.nn as nn
import warnings

def quantize_per_channel(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantization with ultra-low bit support."""
    if bits >= 16:
        return W.detach().clone()
    
    # ── 1-BIT SPECIAL CASE (Sign-only) ──────────────────────────────────
    if bits == 1:
        scale = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale

    # ── MULTI-BIT SYMMETRIC ──────────────────────────────────────────
    # Levels = 2^(bits-1) - 1.  (e.g. 2 bits -> levels=1 -> [-1, 0, 1])
    max_val = max(1, 2 ** (bits - 1) - 1)
    scale = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / max_val
    return (W / scale).round().clamp(-max_val - 1, max_val) * scale

def quantize_shared_scale(W: torch.Tensor, leader_row: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantization using a Shared-Scale."""
    if bits >= 16:
        return W.detach().clone()
    
    if bits == 1:
        # Use leader's average magnitude as the shared scale
        scale = leader_row.abs().mean().clamp(min=1e-8)
        return torch.sign(W) * scale

    max_val = max(1, 2 ** (bits - 1) - 1)
    # Leader's scale
    scale = leader_row.abs().max().clamp(min=1e-8) / max_val
    
    scaled_W = W / scale
    # Clip extreme outliers to prevent numerical explosion
    if (scaled_W.abs().max() > max_val * 4.0):
        warnings.warn(f"[CSAQ] Outlier detected (bits={bits}). Clipping follower weights to 4x leader range.")
        scaled_W = scaled_W.clamp(-max_val * 4.0, max_val * 4.0)
        
    return scaled_W.round().clamp(-max_val - 1, max_val) * scale
