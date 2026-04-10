# CSAQ: Causal Salience-Aware Quantization

**Causal Salience-Aware Quantization (CSAQ)** is a high-performance LLM weight quantization engine designed to hit perfectly defined fractional bit-budgets (e.g., exactly 4.0 bits/weight) by utilizing mixed-precision formats. Unlike magnitude-based proxies like AWQ or GPTQ, CSAQ uses first-order Taylor approximations to measure actual causal salience combined with advanced co-activation interaction graphs.

## Features

- **Multi-Bit Mixed Precision**: Replaces static quantization settings. Automatically distributes available bit thresholds (1, 2, 4, 8, 16) based heavily on impact, significantly minimizing degradation on critical model pathways.
- **Top-K Jaccard Co-Activation Graphs**: Discovers sets of weights that commonly fire together using "Atomic Cliques".
- **Shared-Scale Architecture**: Assigns low-precision bits to trailing follower weights by recycling the Quantization Scaling Factors (S) and Zero-Points (Z) of the clique's high-salience *Leader*, aggressively compressing parameters without losing scale context.
- **Constant Memory Footprint**: Tracks Jaccard activation sparsification using an online bit-vector union/intersection accumulator, avoiding disastrous Out-Of-Memory (OOM) errors during calibration.

## Installation

Install using pip:

```bash
pip install csaq-quant
```

## Quick Start

### Python API

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
```

## License

MIT License
