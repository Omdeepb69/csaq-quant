"""
csaq — Causal Salience-Aware Quantization
==========================================

Quick start::

    from csaq import quantize, CSAQConfig, build_calibration_data

    config = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16])
    calib  = build_calibration_data(tokenizer, n=64)
    model, info = quantize(model, calib, config=config)

    # Self-Speculative Decoding
    from csaq import CSAQInferenceEngine
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer)
    output, report = engine.generate(input_ids, max_new_tokens=256,
                                      speculative=True, lookahead=4)
    print(report.summary())
"""

from .config import CSAQConfig
from .core import (
    CausalProfiler,
    apply_csaq,
    quantize,
    solve_clique_budget,
)
from .inference import CSAQInferenceEngine, SpeculativeReport
from .kernels import CSAQLinear, QuantizedWeight
from .utils import (
    build_calibration_data,
    compute_perplexity,
    export_csaq_model,
    generate_csaq_report,
)
from .modeling import CSAQModelForCausalLM
from transformers import AutoModelForCausalLM

AutoModelForCausalLM.register(CSAQConfig, CSAQModelForCausalLM)

__version__ = "0.5.1"
__author__ = "Omdeep Borkar"
__email__ = "omdeepborkar@gmail.com"

__all__ = [
    # Core pipeline
    "quantize",
    "CSAQConfig",
    "CausalProfiler",
    "solve_clique_budget",
    "apply_csaq",
    # Inference
    "CSAQInferenceEngine",
    "SpeculativeReport",
    # Kernels
    "CSAQLinear",
    "QuantizedWeight",
    # Utils
    "build_calibration_data",
    "compute_perplexity",
    "generate_csaq_report",
    "export_csaq_model",
]
