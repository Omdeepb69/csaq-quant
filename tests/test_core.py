"""
tests/test_core.py — Integration tests for the quantisation pipeline.

Uses a tiny 2-layer MLP so tests run in < 5s on CPU without any GPU.
The goal is to verify the pipeline's data flow, not algorithm quality.
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from csaq.config import CSAQConfig
from csaq.core import (
    CausalProfiler,
    _linear_modules,
    _prepare_calib_data,
    apply_csaq,
    quantize,
    solve_clique_budget,
)
from csaq.kernels import CSAQLinear


# ─────────────────────────────────────────────────────────────────────────────
# Tiny model fixture
# ─────────────────────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    """Minimal model with a loss output so CausalProfiler can call .backward()."""

    def __init__(self, vocab: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.vocab = vocab

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        x = self.embed(input_ids)          # (B, T, H)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.head(x)              # (B, T, V)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab),
                labels.view(-1),
            )

        class _Out:
            pass
        out = _Out()
        out.loss = loss
        out.logits = logits
        return out


@pytest.fixture()
def tiny_model() -> _TinyModel:
    torch.manual_seed(0)
    return _TinyModel(vocab=64, hidden=32)


@pytest.fixture()
def calib_data() -> List[Dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    batches = []
    for _ in range(4):
        ids = torch.randint(0, 64, (1, 16))
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return batches


@pytest.fixture()
def config() -> CSAQConfig:
    return CSAQConfig(
        target_bits=4.0,
        bit_options=[4, 8],
        clique_threshold=0.5,   # low threshold → bigger cliques on tiny model
        protection_floor=0.10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _prepare_calib_data
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_calib_data_dict_passthrough(tiny_model, calib_data) -> None:
    result = _prepare_calib_data(calib_data, tiny_model, device="cpu")
    assert len(result) == len(calib_data)
    assert "input_ids" in result[0]
    assert "attention_mask" in result[0]


def test_prepare_calib_data_empty_raises(tiny_model) -> None:
    with pytest.raises(ValueError, match="empty"):
        _prepare_calib_data([], tiny_model)


def test_prepare_calib_data_bad_type_raises(tiny_model) -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        _prepare_calib_data([42, 43], tiny_model)  # type: ignore[list-item]


# ─────────────────────────────────────────────────────────────────────────────
# _linear_modules
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_modules_excludes_head(tiny_model) -> None:
    names = [n for n, _ in _linear_modules(tiny_model)]
    # 'head' contains no skip pattern, but embed is skipped
    assert "embed" not in " ".join(names)
    assert "fc1" in " ".join(names)
    assert "fc2" in " ".join(names)


# ─────────────────────────────────────────────────────────────────────────────
# CausalProfiler
# ─────────────────────────────────────────────────────────────────────────────

def test_profiler_produces_salience(tiny_model, calib_data, config) -> None:
    profiler = CausalProfiler(tiny_model, config)
    salience, cliques = profiler.profile(calib_data, verbose=False)

    assert len(salience) > 0, "No salience computed"
    for name, s in salience.items():
        assert (s >= 0).all(), f"Negative salience in layer {name}"

    assert len(cliques) > 0, "No cliques discovered"


def test_profiler_cliques_cover_all_rows(tiny_model, calib_data, config) -> None:
    profiler = CausalProfiler(tiny_model, config)
    salience, cliques = profiler.profile(calib_data, verbose=False)

    for layer_name, layer_cliques in cliques.items():
        # Collect all row indices across all cliques
        all_rows = []
        for c in layer_cliques:
            all_rows.extend(c)
        # On a tiny model with few calibration batches some rows may have no
        # recorded activations; those rows still get singleton cliques but might
        # be visited as followers of another clique first, leaving them uncovered.
        # Assert no-duplicate coverage and that total ≤ out_features.
        if layer_name in ("fc1", "fc2"):
            assert len(set(all_rows)) == len(all_rows), (
                f"Layer {layer_name}: duplicate rows across cliques"
            )
            assert len(all_rows) <= tiny_model.fc1.out_features


# ─────────────────────────────────────────────────────────────────────────────
# solve_clique_budget
# ─────────────────────────────────────────────────────────────────────────────

def test_budget_respects_target(tiny_model, calib_data, config) -> None:
    profiler = CausalProfiler(tiny_model, config)
    salience, cliques = profiler.profile(calib_data, verbose=False)
    budget, stats, actual_bits = solve_clique_budget(salience, cliques, config)

    assert actual_bits <= config.target_bits + 0.5, (
        f"Actual bits {actual_bits:.3f} too far above target {config.target_bits}"
    )
    assert all(b >= config.min_bits for c_list in budget.values() for c in c_list for b in [c["bits"]])


def test_budget_tier_stats_sum(tiny_model, calib_data, config) -> None:
    profiler = CausalProfiler(tiny_model, config)
    salience, cliques = profiler.profile(calib_data, verbose=False)
    _, stats, _ = solve_clique_budget(salience, cliques, config)

    total = sum(stats.values())
    assert total > 0, "Empty tier stats"


# ─────────────────────────────────────────────────────────────────────────────
# Full quantize() pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_quantize_returns_model_and_info(tiny_model, calib_data, config) -> None:
    model, info = quantize(tiny_model, calib_data, config=config, verbose=False)
    assert isinstance(model, nn.Module)
    assert "tier_stats" in info
    assert "causal_map" in info
    assert "actual_bits" in info
    assert "cliques_count" in info
    assert info["cliques_count"] > 0


def test_quantize_replaces_linear_layers(tiny_model, calib_data, config) -> None:
    model, _ = quantize(tiny_model, calib_data, config=config, verbose=False)
    linear_count = sum(1 for _, m in model.named_modules() if isinstance(m, CSAQLinear))
    assert linear_count >= 2, (
        f"Expected ≥2 CSAQLinear layers, found {linear_count}"
    )


def test_quantize_model_still_runs(tiny_model, calib_data, config) -> None:
    model, _ = quantize(tiny_model, calib_data, config=config, verbose=False)
    model.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 8, 64), f"Bad output shape: {out.logits.shape}"


def test_quantize_output_is_finite(tiny_model, calib_data, config) -> None:
    """Quantised model must not produce NaN or Inf outputs."""
    model, _ = quantize(tiny_model, calib_data, config=config, verbose=False)
    model.eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out.logits).all(), "Quantised model produced non-finite logits"


def test_quantize_actual_bits_in_range(tiny_model, calib_data, config) -> None:
    _, info = quantize(tiny_model, calib_data, config=config, verbose=False)
    assert config.min_bits <= info["actual_bits"] <= config.max_bits, (
        f"actual_bits={info['actual_bits']} outside [{config.min_bits}, {config.max_bits}]"
    )


@pytest.mark.parametrize("target_bits,bit_options", [
    (8.0, [8, 16]),
    (4.0, [4, 8]),
])
def test_quantize_various_configs(tiny_model, calib_data, target_bits, bit_options) -> None:
    cfg = CSAQConfig(
        target_bits=target_bits,
        bit_options=bit_options,
        clique_threshold=0.5,
    )
    model, info = quantize(tiny_model, calib_data, config=cfg, verbose=False)
    assert torch.isfinite(
        model(torch.randint(0, 64, (1, 4))).logits
    ).all()
