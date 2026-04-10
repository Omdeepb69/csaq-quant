"""
csaq — Causal Salience-Aware Quantization
====================================
pip install csaq-quant

Usage:
    from csaq import quantize, CSAQConfig

    config = CSAQConfig(target_bits=4.0)
    model, info = quantize(model, calib_data, config=config)

    # Self-Speculative Decoding
    from csaq import CSAQInferenceEngine
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer)
    output, report = engine.generate_speculative(input_ids, lookahead=4)
"""

from .config import CSAQConfig
from .core import (
    quantize,
    CausalProfiler,
    solve_clique_budget,
    apply_csaq,
)

from .inference import (
    CSAQInferenceEngine,
    SpeculativeReport,
)

from .utils import (
    build_calibration_data,
    compute_perplexity,
    generate_csaq_report,
    export_csaq_model,
)

__version__ = "0.3.8"
__author__  = "Omdeep Borkar"
__all__     = [
    "quantize",
    "CSAQConfig",
    "CausalProfiler",
    "solve_clique_budget",
    "apply_csaq",
    "CSAQInferenceEngine",
    "SpeculativeReport",
    "build_calibration_data",
    "compute_perplexity",
    "generate_csaq_report",
    "export_csaq_model",
]

