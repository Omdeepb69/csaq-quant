"""
csaq — Causal Salience-Aware Quantization
====================================
pip install csaq-quant

Usage:
    from csaq import quantize, CSAQConfig

    config = CSAQConfig(target_bits=4.0)
    model, info = quantize(model, calib_data, config=config)

Paper: "CSAQ: Causal Salience-Aware Quantization"
"""

from .config import CSAQConfig
from .core import (
    quantize,
    CausalProfiler,
    solve_clique_budget,
    apply_csaq,
)

from .utils import (
    build_calibration_data,
    compute_perplexity,
    generate_csaq_report
)

__version__ = "0.1.0"
__author__  = "Omdeep Borkar"
__all__     = [
    "quantize",
    "CSAQConfig",
    "CausalProfiler",
    "solve_clique_budget",
    "apply_csaq",
    "build_calibration_data",
    "compute_perplexity",
    "generate_csaq_report"
]
