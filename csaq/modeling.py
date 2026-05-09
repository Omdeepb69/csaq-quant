"""
csaq/modeling.py — HuggingFace AutoModelForCausalLM integration.
"""

import json
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download

from .config import CSAQConfig
from .kernels import CSAQLinear, _get_submodule

class CSAQModelForCausalLM(PreTrainedModel):
    config_class = CSAQConfig
    base_model_prefix = "model"

    def __init__(self, config: CSAQConfig):
        super().__init__(config)
        self.csaq_config = config
        
        if not getattr(config, "base_model_name_or_path", None):
            raise ValueError("[CSAQ] base_model_name_or_path is missing in config.")

        base_config = AutoConfig.from_pretrained(
            config.base_model_name_or_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_config(base_config)

        # Replace linear layers with empty CSAQLinear layers matching the bit layout
        # stored in csaq_manifest.json. This allows the subsequent state_dict load
        # to populate the weight_packed and other buffers.
        manifest_path = os.path.join(config.name_or_path, "csaq_manifest.json")
        if not os.path.exists(manifest_path):
            try:
                manifest_path = hf_hub_download(config.name_or_path, "csaq_manifest.json")
            except Exception as e:
                raise FileNotFoundError(f"[CSAQ] csaq_manifest.json not found locally or on Hub at {config.name_or_path}.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        layer_bits = manifest.get("layer_bits", {})
        if not layer_bits:
            raise ValueError("[CSAQ] No layer_bits mapping found in csaq_manifest.json.")

        replaced = 0
        for name, bits in layer_bits.items():
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model if not parent_name else _get_submodule(self.model, parent_name)
            if parent is None:
                continue
            child = getattr(parent, child_name, None)
            if not isinstance(child, nn.Linear):
                continue
            
            csaq_layer = CSAQLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bits=bits,
                bias=child.bias,
                group_size=config.group_size
            )
            setattr(parent, child_name, csaq_layer)
            replaced += 1

        self.all_tied_weights_keys = {}
        if hasattr(self.model, "all_tied_weights_keys"):
            for k, v in self.model.all_tied_weights_keys.items():
                self.all_tied_weights_keys[f"model.{k}"] = f"model.{v}"

        # We don't print here to keep from_pretrained quiet, but we successfully mapped layers.

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
