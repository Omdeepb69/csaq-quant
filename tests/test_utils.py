"""
tests/test_utils.py — Tests for csaq.utils helpers.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from csaq.config import CSAQConfig
from csaq.utils import export_csaq_model, generate_csaq_report


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.config = type("cfg", (), {"model_type": "tiny"})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def state_dict(self, **kw):  # type: ignore[override]
        return super().state_dict(**kw)


@pytest.fixture()
def tiny_model() -> _TinyModel:
    return _TinyModel()


@pytest.fixture()
def sample_info() -> dict:
    return {
        "tier_stats": {"int4": 512, "int8": 256},
        "actual_bits": 5.33,
        "cliques_count": 12,
        "elapsed_s": 3.14,
        "causal_map": {"fc": [0, 1, 2]},
        "budget": {},
    }


@pytest.fixture()
def config() -> CSAQConfig:
    return CSAQConfig(target_bits=4.0, bit_options=[4, 8])


# ─────────────────────────────────────────────────────────────────────────────
# generate_csaq_report
# ─────────────────────────────────────────────────────────────────────────────

def test_report_creates_file(sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        result = generate_csaq_report(sample_info, save_path=path)
        assert os.path.isfile(path)
        assert isinstance(result, dict)


def test_report_schema(sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        result = generate_csaq_report(sample_info, save_path=path)

        required_keys = {
            "csaq_version",
            "actual_avg_bits",
            "total_cliques",
            "total_quantized_params",
            "bit_distribution",
            "bit_distribution_pct",
        }
        for key in required_keys:
            assert key in result, f"Missing key in report: {key}"


def test_report_bit_distribution_sums_to_100(sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        result = generate_csaq_report(sample_info, save_path=path)
        pct_sum = sum(result["bit_distribution_pct"].values())
        assert abs(pct_sum - 100.0) < 0.01, f"Percentages sum to {pct_sum}"


def test_report_json_valid(sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        generate_csaq_report(sample_info, save_path=path)
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["actual_avg_bits"] == round(sample_info["actual_bits"], 4)


def test_report_empty_tier_stats() -> None:
    """Report must not crash when tier_stats is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        result = generate_csaq_report(
            {"tier_stats": {}, "actual_bits": 0.0, "cliques_count": 0},
            save_path=path,
        )
        # When tier_stats is empty the report still writes correctly
        assert result["bit_distribution"] == {}
        assert result["bit_distribution_pct"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# export_csaq_model
# ─────────────────────────────────────────────────────────────────────────────

def test_export_creates_expected_files(tiny_model, config, sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)

        assert os.path.isfile(os.path.join(tmpdir, "config.json"))
        assert os.path.isfile(os.path.join(tmpdir, "csaq_manifest.json"))
        assert os.path.isfile(os.path.join(tmpdir, "model.safetensors"))


def test_export_config_json_has_quant_config(tiny_model, config, sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)

        with open(os.path.join(tmpdir, "config.json"), encoding="utf-8") as f:
            cfg_dict = json.load(f)

        assert "quantization_config" in cfg_dict
        assert cfg_dict["quantization_config"]["quant_type"] == "csaq"
        assert cfg_dict["quantization_config"]["target_bits"] == 4.0


def test_export_manifest_schema(tiny_model, config, sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)

        with open(os.path.join(tmpdir, "csaq_manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)

        assert "csaq_version" in manifest
        assert "bit_distribution" in manifest
        assert "actual_avg_bits" in manifest
        assert "causal_map" in manifest


def test_export_safetensors_loadable(tiny_model, config, sample_info) -> None:
    """The exported safetensors file must be loadable with safetensors.torch."""
    import safetensors.torch

    with tempfile.TemporaryDirectory() as tmpdir:
        export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)
        st_path = os.path.join(tmpdir, "model.safetensors")
        loaded = safetensors.torch.load_file(st_path)
        assert "fc.weight" in loaded
        assert loaded["fc.weight"].shape == (4, 8)


def test_export_excludes_csaq_buffers(tiny_model, config, sample_info) -> None:
    """Non-persistent CSAQ speculative buffers must NOT appear in safetensors."""
    import safetensors.torch

    # Register dummy non-persistent buffers to simulate quantised model
    tiny_model.fc.register_buffer(
        "_csaq_fp16_backup", torch.zeros(2, 8), persistent=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)
        loaded = safetensors.torch.load_file(os.path.join(tmpdir, "model.safetensors"))
        for key in loaded:
            assert "_csaq_" not in key, f"Unexpected CSAQ buffer in export: {key}"


def test_export_return_value(tiny_model, config, sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_csaq_model(tiny_model, config, {}, tmpdir, info=sample_info)
        assert os.path.isabs(result)
        assert os.path.isdir(result)


def test_export_creates_dir_if_missing(tiny_model, config, sample_info) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "nested", "output")
        export_csaq_model(tiny_model, config, {}, new_dir, info=sample_info)
        assert os.path.isdir(new_dir)
