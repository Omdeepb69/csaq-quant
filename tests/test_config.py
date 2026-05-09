"""
tests/test_config.py — Unit tests for CSAQConfig validation.
"""

from __future__ import annotations

import pytest

from csaq.config import CSAQConfig


def test_default_config() -> None:
    cfg = CSAQConfig()
    assert cfg.target_bits == 4.0
    assert cfg.bit_options == [4, 8, 16]
    assert cfg.clique_threshold == 0.85


def test_valid_custom_config() -> None:
    cfg = CSAQConfig(target_bits=2.5, bit_options=[2, 4], clique_threshold=0.75)
    assert cfg.target_bits == 2.5
    assert cfg.bit_options == [2, 4]


def test_invalid_bit_options_filtered() -> None:
    """1-bit and 3-bit should be silently filtered out."""
    with pytest.warns(UserWarning, match="unsupported widths"):
        cfg = CSAQConfig(target_bits=4.0, bit_options=[1, 4, 8])
    assert 1 not in cfg.bit_options
    assert 4 in cfg.bit_options


def test_no_valid_options_raises() -> None:
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="No valid bit_options"):
            CSAQConfig(target_bits=1.0, bit_options=[1, 3])


def test_target_bits_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="target_bits"):
        CSAQConfig(target_bits=3.0, bit_options=[4, 8])   # 3.0 < min(4)


def test_clique_threshold_bounds() -> None:
    with pytest.raises(ValueError, match="clique_threshold"):
        CSAQConfig(clique_threshold=-0.1)
    with pytest.raises(ValueError, match="clique_threshold"):
        CSAQConfig(clique_threshold=1.5)


def test_protection_floor_bounds() -> None:
    with pytest.raises(ValueError, match="protection_floor"):
        CSAQConfig(protection_floor=-0.1)
    with pytest.raises(ValueError, match="protection_floor"):
        CSAQConfig(protection_floor=1.0)


def test_min_max_bits_properties() -> None:
    cfg = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16])
    assert cfg.min_bits == 4
    assert cfg.max_bits == 16


def test_bit_options_sorted() -> None:
    cfg = CSAQConfig(target_bits=4.0, bit_options=[16, 4, 8])
    assert cfg.bit_options == [4, 8, 16]


def test_group_size_stored() -> None:
    cfg = CSAQConfig(group_size=128)
    assert cfg.group_size == 128


def test_hf_to_dict_round_trip() -> None:
    """CSAQConfig extends PretrainedConfig — to_dict must survive a round-trip."""
    cfg = CSAQConfig(target_bits=4.0, bit_options=[4, 8])
    d = cfg.to_dict()
    assert d["target_bits"] == 4.0
    assert d["bit_options"] == [4, 8]
    assert d["model_type"] == "csaq"
