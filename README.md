# CSAQ: Causal Salience-Aware Quantization

[![PyPI version](https://badge.fury.io/py/csaq-quant.svg)](https://badge.fury.io/py/csaq-quant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Causal Salience-Aware Quantization (CSAQ)** is a high-performance LLM weight quantization engine designed to hit perfectly defined fractional bit-budgets (e.g., exactly 4.0 bits/weight) by utilizing mixed-precision formats. Unlike magnitude-based proxies like AWQ or GPTQ, CSAQ uses first-order Taylor approximations to measure actual *causal salience* combined with advanced *co-activation interaction graphs*.

## Features

- **Multi-Bit Mixed Precision**: Replaces static quantization settings. Automatically distributes available bit thresholds (`1, 2, 4, 8, 16`) based heavily on impact, significantly minimizing degradation on critical model pathways.
- **Top-K Jaccard Co-Activation Graphs**: Discovers sets of weights that commonly fire together using "Atomic Cliques".
- **Shared-Scale Architecture**: Assigns low-precision bits to trailing follower weights by recycling the Quantization Scaling Factors ($S$) and Zero-Points ($Z$) of the clique's high-salience *Leader*, aggressively compressing parameters without losing scale context.
- **Constant Memory Footprint**: Tracks Jaccard activation sparsification using an online bit-vector union/intersection accumulator, avoiding disastrous Out-Of-Memory (OOM) errors during calibration.

## Installation

Install using pip:

```bash
pip install csaq-quant
```

## Quick Start

### 1. Python API

You can programmatically apply CSAQ using the core export `quantize` and managing constraints with `CSAQConfig`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csaq import quantize, CSAQConfig, build_calibration_data

# 1. Load your standard HF LLM
model_id = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# 2. Extract representative calibration data
calib_data = build_calibration_data(tokenizer, n=32, seq_len=128)

# 3. Configure fractional Bit-Budget and allowed bits (e.g., target exactly 4 bits on average)
config = CSAQConfig(
    target_bits=4.0, 
    bit_options=[1, 2, 4, 8, 16],
    clique_threshold=0.85
)

# 4. Fire the Quantization Pipeline
quantized_model, info = quantize(
    model=model, 
    calib_data=calib_data, 
    config=config, 
    verbose=True
)

# Optional: Generate metrics report and export to safetensors
from csaq.utils import generate_csaq_report, export_csaq_model
generate_csaq_report(info, save_path="./CSAQ_Report.json")
export_csaq_model(quantized_model, config, info["budget"], "./csaq_output")
```

### 2. Command Line Interface (CLI)

CSAQ includes a seamless CLI for quantizing models directly from the terminal without writing a single script.

**Basic Usage:**
```bash
python -m csaq \
  --model_path Qwen/Qwen1.5-0.5B \
  --wbits 4.0 \
  --options 1,2,4,8,16 \
  --save_path ./csaq_export
```

**CLI Arguments Breakdown:**
- `--model_path`: **(Required)** The path to a local Hugging Face model directory or a Hub repository ID (e.g., `meta-llama/Llama-3-8B`).
- `--wbits`: **(Optional)** The target *average* bit-width per weight across the entire network. Defaults to `4.0`.
- `--options`: **(Optional)** Comma-separated list of discrete bit formats the constraint solver is allowed to assign. Defaults to `1,2,4,8,16`. High-salience cliques get higher bits.
- `--save_path`: **(Required)** The local directory where the `safetensors` model, config modifications, and `CSAQ_Report.json` will be saved.

**Example for deep quantization:**
```bash
python -m csaq --model_path meta-llama/Llama-3-8B --wbits 2.5 --options 1,2,4 --save_path ./llama3-2.5bit
```

### Advanced Details

**Early Stopping Heuristic**: The profiling phase evaluates the Spearman rank correlation of accumulated salience gradients over calibration batches. CSAQ will automatically stop processing subsets once weights theoretically stabilize, saving enormous amounts of compute.

**Outputs**: The engine spits out a `CSAQ_Report.json` providing metric insights regarding constraint mapping, Bit-Distribution Histograms, Pareto Efficiency estimation, and Clique generation sizes. 
 
## License

MIT License
