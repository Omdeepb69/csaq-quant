# csaq-quant

**Causal Salience-Aware Quantization** for large language models.

[![CI](https://github.com/omdeepb69/csaq-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/omdeepb69/csaq-quant/actions)
[![PyPI](https://img.shields.io/pypi/v/csaq-quant.svg)](https://pypi.org/project/csaq-quant/)
[![Python](https://img.shields.io/pypi/pyversions/csaq-quant.svg)](https://pypi.org/project/csaq-quant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is CSAQ?

CSAQ is a post-training quantization (PTQ) library that assigns different bit-widths to different weight groups based on how much each group influences the model's output — its *causal salience*.

The core algorithm has two steps:

**Step 1 — Salience scoring.**
Run calibration data through the model with gradients enabled.  For each weight element, compute `|∂L/∂w × w|` (gradient × weight magnitude) — a first-order Taylor approximation of how much removing that weight would change the loss.  This is the same sensitivity measure used in GPTQ/OBC.

**Step 2 — Clique grouping.**
Track which output channels activate together across samples (Jaccard co-activation similarity).  Channels that reliably fire together form a *clique* and share a quantisation scale.  This reduces per-parameter metadata overhead while keeping semantically related weights at the same precision.

High-salience cliques receive more bits; low-salience followers receive fewer bits.  A greedy solver distributes bits to hit the target average bit-width.

The same clique structure powers **self-speculative decoding**: high-salience rows are backed up in fp16 for the verify pass, while the draft pass uses fully-quantised weights — giving a speedup with near-zero extra memory.

> **Status:** Alpha (v0.5.1).  The algorithm is implemented and runs end-to-end.
> Perplexity benchmarks vs GPTQ/AWQ are in progress — see [Benchmarks](#benchmarks).

---

## Installation

```bash
pip install csaq-quant
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.38.

Optional extras:
```bash
pip install "csaq-quant[dev]"    # pytest, ruff, mypy, black
pip install "csaq-quant[triton]" # Triton kernel support (planned)
```

---

## Quick start

### Quantise a model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from csaq import quantize, CSAQConfig, build_calibration_data

model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

# Build calibration data (64 WikiText-2 samples, seq_len=128)
calib = build_calibration_data(tokenizer, n=64, seq_len=128)

# Configure: target 4-bit average, allow 4/8/16-bit assignment per clique
config = CSAQConfig(
    target_bits=4.0,
    bit_options=[4, 8, 16],
    clique_threshold=0.85,  # Jaccard threshold for grouping channels
    protection_floor=0.10,  # always keep top 10% salient rows at ≥8-bit
    group_size=128,         # per-group scales (better accuracy at low bits)
)

model, info = quantize(model, calib, config=config)

print(f"Actual avg bits: {info['actual_bits']:.3f}")
print(f"Cliques discovered: {info['cliques_count']}")
print(f"Bit distribution: {info['tier_stats']}")
```

### Run inference

```python
# Standard inference — the quantised model is a normal nn.Module
model.eval()
inputs = tokenizer("The capital of France is", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### Self-Speculative Decoding

```python
from csaq import CSAQInferenceEngine

engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer)

input_ids = tokenizer("The theory of relativity", return_tensors="pt").input_ids
output, report = engine.generate(
    input_ids,
    speculative=True,
    lookahead=4,          # draft tokens per block
    max_new_tokens=200,
    temperature=0.8,
)

print(tokenizer.decode(output[0]))
print(report.summary())
# {'acceptance_rate': 0.72, 'speedup_factor': 2.1, 'inter_token_latency_ms': 18.3, ...}
```

### Save and reload

```python
from csaq.utils import export_csaq_model, generate_csaq_report
from transformers import AutoModelForCausalLM

# Save — writes config.json + csaq_manifest.json + model.safetensors
export_csaq_model(model, config, info["budget"], "./my-model-4bit", info=info)

# Save a JSON report
generate_csaq_report(info, save_path="./my-model-4bit/CSAQ_Report.json")

# Reload — AutoModelForCausalLM seamlessly reinstantiates the quantised architecture
reloaded_model = AutoModelForCausalLM.from_pretrained("./my-model-4bit", device_map="auto")
```

### Domain-specific calibration

```python
# Supply your own calibration texts for better domain accuracy
calib = build_calibration_data(
    tokenizer,
    n=128,
    custom_texts=my_domain_sentences,   # list of strings
)
```

---

## CLI

```bash
# Basic 4-bit quantisation
python -m csaq \
    --model_path Qwen/Qwen1.5-0.5B \
    --wbits 4.0 \
    --save_path ./qwen-4bit

# GPU, hard calibration, post-quant PPL
python -m csaq \
    --model_path meta-llama/Llama-3-8B \
    --wbits 4.0 --options 4,8,16 \
    --group_size 128 \
    --hard_calib \
    --eval_ppl \
    --device auto \
    --save_path ./llama3-4bit

# Show all options
python -m csaq --help
```

---

## Configuration reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_bits` | `float` | `4.0` | Target average bits-per-weight |
| `bit_options` | `List[int]` | `[4, 8, 16]` | Allowed bit widths (2/4/8/16 only) |
| `clique_threshold` | `float` | `0.85` | Jaccard similarity threshold for clique grouping |
| `protection_floor` | `float` | `0.10` | Fraction of salient rows always kept at ≥8-bit |
| `group_size` | `int` | `-1` | Per-group scale granularity (-1 = per-channel) |
| `salience_alpha` | `float` | `1.0` | Activation sparsity mask scaling factor |
| `speculative_lookahead` | `int` | `4` | Default draft tokens per speculative block |

**Supported bit widths:** `2`, `4`, `8`, `16`.
1-bit is not supported — sign-only quantization causes catastrophic accuracy loss in LLMs and has no matching PyTorch storage type.

---

## Benchmarks

> Benchmarks are in progress.  Run them yourself with:

```bash
python benchmarks/benchmark_ppl.py \
    --model_path Qwen/Qwen1.5-0.5B \
    --bit_configs "4.0:4,8,16" "3.0:4,8" \
    --n_calib 64

python benchmarks/benchmark_speculative.py \
    --model_path Qwen/Qwen1.5-0.5B \
    --lookahead 4 6 8
```

Planned comparison targets: GPTQ (AutoGPTQ), AWQ (AutoAWQ), HQQ — all evaluated on WikiText-2 PPL with `stride=512, max_tokens=4096`.

**Example Output:**
```text
─────────────────────────────────────────────────────────────────────────────────────
Config                          Bits       PPL    vs FP32       VRAM saved     Time
─────────────────────────────────────────────────────────────────────────────────────
FP32 baseline                  32.00    12.345          —                —     1.2s
CSAQ 4.0-bit [4,8,16]           4.01    12.450      +0.8%   1.45GB (87.5%)     3.4s
CSAQ 3.0-bit [4,8]              3.05    13.100      +6.1%   1.62GB (90.2%)     3.2s
─────────────────────────────────────────────────────────────────────────────────────
```

---

## How it compares to GPTQ / AWQ

| Feature | GPTQ | AWQ | **CSAQ** |
|---------|------|-----|---------|
| Salience metric | Hessian (OBC) | Activation scale | Gradient × weight |
| Weight grouping | Per-channel | Per-channel | **Jaccard cliques** |
| Mixed precision | Manual | Yes | **Automatic** |
| Self-speculative decoding | No | No | **Yes** |
| Group quantisation | Yes (128) | Yes | Yes |
| Supported bits | 2/3/4/8 | 4/8 | 2/4/8/16 |
| PPL benchmarks | ✅ Published | ✅ Published | 🔄 In progress |

CSAQ's clique-based grouping is novel: instead of treating each output channel independently, channels that co-activate are quantised with a shared scale.  This reduces metadata overhead and allows the speculative decoding engine to work with the natural "salience topology" of the model.

---

## Speculative decoding

CSAQ includes a built-in **self-speculative decoding** engine: the quantised model acts as its own draft model, with high-salience rows swapped to fp16 for the verify pass. No separate draft model is needed.

```python
from csaq.inference import CSAQInferenceEngine

engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer)
output, report = engine.generate(input_ids, speculative=True, lookahead=4, max_new_tokens=128)
print(report.summary())
```

**Benchmark results** *(placeholder — run on your own hardware for accurate numbers)*:

```text
Model: Qwen/Qwen1.5-0.5B  |  Bits: 4.12  |  Device: cuda
┌──────────────────────┬────────────┬──────────┬─────────┐
│ Mode                 │ Accept     │ tok/s    │ Speedup │
├──────────────────────┼────────────┼──────────┼─────────┤
│ Standard (quantised) │ —          │ 42.1     │ 1.00×   │
│ Speculative la=4     │ 68.3%      │ 71.4     │ 1.70×   │
│ Speculative la=6     │ 65.1%      │ 79.2     │ 1.88×   │
│ Speculative la=8     │ 61.8%      │ 81.3     │ 1.93×   │
└──────────────────────┴────────────┴──────────┴─────────┘
```

> Run `benchmarks/validate_speculative.py` to generate your model's numbers:
> ```bash
> python benchmarks/validate_speculative.py \
>     --model_path Qwen/Qwen1.5-0.5B \
>     --calib_file calib.txt \
>     --output_path speculative_results.json
> ```

---

## Development

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest                           # all fast tests
pytest -m "not slow"             # exclude slow tests
pytest tests/test_kernels.py -v  # specific module

# Run linting
ruff check csaq/
black csaq/
mypy csaq/
```

---

## Project structure

```
csaq-quant/
├── csaq/
│   ├── __init__.py       Public API exports
│   ├── config.py         CSAQConfig — validated, HF-compatible
│   ├── kernels.py        Bit-packing, QuantizedWeight, CSAQLinear
│   ├── core.py           Three-phase pipeline (profile → solve → apply)
│   ├── inference.py      CSAQInferenceEngine, self-speculative decoding
│   ├── utils.py          Calibration, PPL evaluation, export
│   └── __main__.py       CLI entry point
├── tests/
│   ├── test_kernels.py   Pack/unpack round-trips, CSAQLinear
│   ├── test_config.py    Config validation
│   ├── test_core.py      Full pipeline integration tests
│   ├── test_inference.py Speculative decoding engine
│   └── test_utils.py     Calibration, export, reporting
├── benchmarks/
│   ├── benchmark_ppl.py          WikiText-2 PPL comparison
│   └── benchmark_speculative.py  Speculative decoding throughput
├── .github/workflows/ci.yml
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Roadmap

- [ ] **Publish PPL benchmark table** — Qwen1.5-0.5B, Llama-3-8B at 2/4/8-bit vs GPTQ/AWQ
- [ ] **Triton dequant kernel** — replace pure-Python `_unpack()` with a fused Triton op for ~2× faster inference
- [ ] **HuggingFace `from_pretrained` loader** — register `CSAQConfig` in the AutoClass mapping so quantised models reload in one line
- [ ] **`lm-eval-harness` integration** — zero-shot accuracy on ARC, HellaSwag, MMLU
- [ ] **GGUF / llama.cpp export** — for CPU inference use cases

---

## Citation

If you use CSAQ in research, please cite:

```bibtex
@software{borkar2024csaq,
  author  = {Borkar, Omdeep},
  title   = {{CSAQ}: Causal Salience-Aware Quantization},
  year    = {2024},
  url     = {https://github.com/omdeepb69/csaq-quant},
  version = {0.5.1}
}
```

---

## License

MIT © Omdeep Borkar
