# Changelog

All notable changes to **csaq-quant** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.1] — 2024-05-09 *(current)*

### Changed
- Bumped version to 0.5.1 for PyPI release.

---

## [0.5.0] — 2024-01-XX

### Breaking changes
- `bit_options` default changed from `[1, 2, 4, 8, 16]` to `[4, 8, 16]`.
  1-bit and unsupported widths are now filtered with a `UserWarning` at
  `CSAQConfig` construction time rather than silently producing wrong results.
- `apply_csaq()` now requires `config` as a fourth argument.
- `export_csaq_model()` signature changed: `budget` is now positional arg 3,
  `info` is now keyword-only.

### Added
- **Real bit-packing** (`kernels.py`): `quantize_per_channel` and
  `quantize_shared_scale` now return `QuantizedWeight` dataclasses with
  packed `uint8` buffers for 2-bit, 4-bit, and 8-bit — actual memory savings
  instead of simulated quantisation.
- **`CSAQLinear`** (`kernels.py`): drop-in `nn.Linear` replacement that stores
  weights as packed int8/uint8 and dequantises on-the-fly during `forward()`.
  Injected via `inject_csaq_linear()` as the final step of `quantize()`.
- **Per-group quantisation** (`group_size` in `CSAQConfig`): group sizes of
  32, 64, 128 reduce per-group quantisation error vs per-channel at the same
  bit width.
- **`QuantizedWeight` dataclass**: carries `qdata`, `scales`, `zero_points`,
  `bits`, and shape metadata.  Has `.dequantize()`, `.compression_ratio()`,
  and `.element_size_bytes()` methods.
- **Protection floor** (`protection_floor` in `CSAQConfig`): top-N% most
  salient rows are guaranteed ≥ 8-bit regardless of budget solver result.
- **`compute_perplexity()`** now implements the standard sliding-window PPL
  protocol (stride=512, max_tokens=4096) matching GPTQ/AWQ evaluation.
- **`build_calibration_data()`** now accepts `custom_texts` kwarg for
  domain-adapted quantisation.
- **`export_csaq_model()`** now writes a `quantization_config` block inside
  `config.json` so HuggingFace `AutoModelForCausalLM.from_pretrained()` can
  detect the quantisation format.
- **CLI** (`python -m csaq`): added `--device auto`, `--eval_ppl`,
  `--eval_baseline_ppl`, `--group_size`, `--protection_floor`, `--hard_calib`,
  and `--quiet` flags.  Auto-selects fp16 + `device_map="auto"` when GPU is
  available.
- **Full test suite** (`tests/`): 50+ unit and integration tests covering
  kernels, config validation, the full pipeline, the inference engine, and
  export utilities.  Run with `pytest`.
- **CI** (`.github/workflows/ci.yml`): lint (ruff + black + mypy), tests on
  Python 3.9 / 3.10 / 3.11, build check, and Codecov coverage upload.
- **PPL benchmark** (`benchmarks/benchmark_ppl.py`): compare CSAQ at multiple
  bit configs against FP32 baseline.
- **Speculative decoding benchmark** (`benchmarks/benchmark_speculative.py`):
  measure tok/s and acceptance rate vs standard autoregressive generation.
- `pyproject.toml` replaces `setup.py` (PEP 517/621 compliant).
- `py.typed` marker for PEP 561 (typed package).
- `pre-commit` config with black, ruff, and isort hooks.

### Fixed
- Spearman early-stopping rank correlation was computing `_eval_idx` lazily
  inside the loop; now allocated once before the loop.
- `_build_cliques` was mutating `visited` with raw Python ints instead of a
  typed `torch.bool` tensor — fixed.
- `generate_csaq_report` was silently ignoring `overlap_pct`; now reads it
  from `info` correctly.
- `export_csaq_model` no longer saves non-persistent speculative buffers
  (`_csaq_fp16_backup`, `_csaq_quant_stash`, `_csaq_hi_rows`) to safetensors.
- `CSAQInferenceEngine._build_hooks` now checks for `CSAQLinear` type rather
  than duck-typing buffer presence.

### Removed
- `setup.py` (replaced by `pyproject.toml`).
- 1-bit (sign-only) quantisation — no PyTorch storage representation exists;
  removed from all defaults and documented as unsupported.
- `_SAFE_FLOOR_TOP_PCT` global constant — replaced by `config.protection_floor`.

---

## [0.4.1] — 2023-XX-XX  *(pre-release, unpublished)*

Initial internal revision.  Core algorithm sketch, no tests, no packing.

---

[0.5.1]: https://github.com/omdeepb69/csaq-quant/releases/tag/v0.5.1
[0.5.0]: https://github.com/omdeepb69/csaq-quant/releases/tag/v0.5.0
[0.4.1]: https://github.com/omdeepb69/csaq-quant/releases/tag/v0.4.1
