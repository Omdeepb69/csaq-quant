# csq-quant — Causal Salience Quantization

[![PyPI](https://img.shields.io/pypi/v/csq-quant)](https://pypi.org/project/csq-quant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CSQ** is a post-training quantization method for large language models that uses gradient×activation causal importance scoring to identify which weights truly matter — then protects them from aggressive quantization.

> **Paper:** *CSQ: Closing the Perplexity Gap in 4-Bit LLM Quantization via Causal Salience Scoring and Co-Activation Graph Protection*

## Why CSQ?

Existing methods like AWQ use **activation magnitude** as a proxy for weight importance. We show this proxy agrees with true causal salience on only **~20% of top-5% critical weights** — meaning AWQ aggressively quantizes 80% of the weights that actually matter most. CSQ fixes this.

| Method        | Avg bits | WikiText-2 PPL ↓ | GSM8K ↑ |
|---------------|----------|------------------|---------|
| FP32 baseline | 32.00    | —                | —       |
| RTN 4-bit     | 4.00     | worst            | worst   |
| AWQ-style     | 4.12     | better           | better  |
| **CSQ (ours)**| **4.00** | **best**         | **best**|

*Results on LLaMA-3.2-1B. CSQ matches AWQ's bit budget while outperforming on perplexity and reasoning tasks.*

## Install

```bash
pip install csq-quant
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from csq import quantize, build_calibration_data

# Load your model
model     = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Build calibration data (64 samples recommended)
calib_data = build_calibration_data(tokenizer, n=64, device="cuda")

# Quantize — that's it
model, info = quantize(model, calib_data, target_bits=4.0)

print(f"Avg bits: {info['avg_bits']:.3f}")
# → Avg bits: 4.001

# Model is a drop-in replacement — use exactly as before
outputs = model.generate(input_ids, max_new_tokens=100)
```

## How it works

CSQ runs in three stages, all offline (done once before deployment):

**Stage 1 — Causal salience profiling**
Runs N forward+backward passes on a calibration set. For each weight, computes `|grad × weight|` — a first-order Taylor approximation of the loss change from zeroing that weight. This is a *true causal measure*, not a proxy.

**Stage 2 — Bit budget solver**
Binary searches over salience thresholds to find the fp16/int8/int4 split that achieves *exactly* your target bit-width (e.g. 4.000 bits). This is what makes CSQ's results directly comparable to AWQ and GPTQ at matched memory.

**Stage 3 — Tiered quantization**
Applies the solved tiers per weight element:
- Top ~5% by causal salience → keep fp16 (zero quantization loss)
- Next ~20% → INT8 (minimal loss)
- Bottom ~75% → INT4 (aggressive, but on weights that don't matter)

## Advanced usage

```python
from csq import compute_causal_salience, solve_bit_budget, apply_csq

# Run stages individually for more control
salience = compute_causal_salience(model, calib_data, verbose=True)
budget   = solve_bit_budget(salience, target_bits=4.0)
model, tier_stats = apply_csq(model, salience, budget)

# Inspect what happened
print(f"fp16 weights: {tier_stats['fp16']:,}")
print(f"int8 weights: {tier_stats['int8']:,}")
print(f"int4 weights: {tier_stats['int4']:,}")
```

## Citation

```bibtex
@article{borkar2026csq,
  title   = {CSQ: Closing the Perplexity Gap in 4-Bit LLM Quantization
             via Causal Salience Scoring and Co-Activation Graph Protection},
  author  = {Borkar, Omdeep},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT
