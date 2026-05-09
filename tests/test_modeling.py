import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

import csaq
from csaq.config import CSAQConfig
from csaq.core import quantize
from csaq.utils import export_csaq_model

from .test_core import _TinyModel, tiny_model, calib_data


def mock_from_config(*args, **kwargs):
    return _TinyModel(vocab=64, hidden=32)


original_from_pretrained = AutoConfig.from_pretrained

def mock_auto_config(pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "tiny_model_path":
        return type("MockConfig", (), {"model_type": "tiny_model"})()
    return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def test_automodel_round_trip(tiny_model: _TinyModel, calib_data: list) -> None:
    config = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16], clique_threshold=0.5)
    
    # 1. Quantise tiny model
    q_model, info = quantize(tiny_model, calib_data, config=config, verbose=False)
    
    # Run once to get baseline logits
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        baseline_logits = q_model(x).logits

    # Mock the config attribute that HF models normally have
    q_model.config = type("Config", (), {
        "model_type": "tiny_model", 
        "_name_or_path": "tiny_model_path",
        "to_dict": lambda self: {"model_type": "tiny_model", "_name_or_path": "tiny_model_path"}
    })()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 2. Export
        export_csaq_model(q_model, config, info["budget"], tmp_dir, info=info)
        
        # 3. Reload
        with patch("csaq.modeling.AutoConfig.from_pretrained", mock_auto_config), \
             patch("csaq.modeling.AutoModelForCausalLM.from_config", mock_from_config):
            reloaded = AutoModelForCausalLM.from_pretrained(tmp_dir)
            
        assert reloaded is not None
        
        # 4. Assert output logits are finite and shape matches
        reloaded.eval()
        with torch.no_grad():
            new_logits = reloaded(x).logits
            
        assert torch.isfinite(new_logits).all()
        assert new_logits.shape == baseline_logits.shape
