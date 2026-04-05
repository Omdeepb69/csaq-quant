import torch
import torch.nn as nn

def quantize_per_channel(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantization."""
    if bits >= 16:
        return W.clone()
    max_val = 2 ** (bits - 1) - 1
    scale = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / max_val
    return (W / scale).round().clamp(-max_val - 1, max_val) * scale

def quantize_shared_scale(W: torch.Tensor, leader_row: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantization using a Shared-Scale."""
    if bits >= 16:
        return W.clone()
    max_val = 2 ** (bits - 1) - 1
    # Leader's scale
    scale = leader_row.abs().max().clamp(min=1e-8) / max_val
    return (W / scale).round().clamp(-max_val - 1, max_val) * scale
