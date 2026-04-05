import torch
from transformers import PretrainedConfig
from typing import List, Optional

class CSAQConfig(PretrainedConfig):
    model_type = "csaq"

    def __init__(
        self,
        target_bits: float = 4.0,
        bit_options: Optional[List[int]] = None,
        clique_threshold: float = 0.85,
        auto_scale_memory: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_bits = target_bits
        self.bit_options = bit_options if bit_options is not None else [1, 2, 4, 8, 16]
        self.clique_threshold = clique_threshold
        self.auto_scale_memory = auto_scale_memory
