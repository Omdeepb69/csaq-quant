from __future__ import annotations

import pytest
import torch

from csaq.config import CSAQConfig
from csaq.core import CausalProfiler, quantize

from .test_core import _TinyModel, tiny_model, calib_data


def test_clique_modes_produce_valid_outputs(tiny_model: _TinyModel, calib_data: list) -> None:
    # ── Run Jaccard mode ──
    cfg_jac = CSAQConfig(clique_mode="jaccard", clique_threshold=0.0)
    profiler = CausalProfiler(tiny_model, cfg_jac)
    _, cliques_jac = profiler.profile(calib_data, verbose=False)
    
    # Check Jaccard produced at least one multi-member clique
    has_multi = any(len(c) > 1 for cl_list in cliques_jac.values() for c in cl_list)
    assert has_multi, "Jaccard mode should produce at least one multi-member clique at threshold 0.0"

    model_jac, info_jac = quantize(tiny_model, calib_data, config=cfg_jac, verbose=False)
    x = torch.randint(0, 64, (1, 8))
    out_jac = model_jac(x).logits
    assert torch.isfinite(out_jac).all(), "Jaccard mode produced non-finite logits"

    # ── Run per_channel mode ──
    # Reload tiny model to ensure clean state
    torch.manual_seed(0)
    tiny_model_pc = _TinyModel(vocab=64, hidden=32)
    
    cfg_pc = CSAQConfig(clique_mode="per_channel")
    profiler_pc = CausalProfiler(tiny_model_pc, cfg_pc)
    _, cliques_pc = profiler_pc.profile(calib_data, verbose=False)
    
    # Check per_channel cliques are all length 1
    all_singletons = all(len(c) == 1 for cl_list in cliques_pc.values() for c in cl_list)
    assert all_singletons, "per_channel mode should only produce cliques of length 1"
    
    model_pc, info_pc = quantize(tiny_model_pc, calib_data, config=cfg_pc, verbose=False)
    out_pc = model_pc(x).logits
    assert torch.isfinite(out_pc).all(), "per_channel mode produced non-finite logits"
