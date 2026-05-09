"""
benchmarks/benchmark_ppl.py — Perplexity benchmark: CSAQ vs FP32 baseline.

Runs WikiText-2 PPL evaluation on a model at multiple bit widths and prints
a comparison table.  Results can be compared directly to published GPTQ/AWQ
numbers using the same stride=512, max_tokens=4096 settings.

Usage::

    python benchmarks/benchmark_ppl.py \\
        --model_path Qwen/Qwen1.5-0.5B \\
        --bit_configs "4.0:4,8,16" "3.0:4,8" "2.5:2,4" \\
        --n_calib 64 --seq_len 128

Output (example)::

    Model: Qwen/Qwen1.5-0.5B
    ┌─────────────────────┬──────────────┬──────────┬────────────┐
    │ Config              │ Actual Bits  │ PPL      │ vs FP32    │
    ├─────────────────────┼──────────────┼──────────┼────────────┤
    │ FP32 baseline       │ 32.00        │ 12.34    │ —          │
    │ CSAQ 4.0-bit [4,8]  │ 4.12         │ 13.01    │ +5.4%      │
    │ CSAQ 3.0-bit [4,8]  │ 3.28         │ 14.55    │ +17.9%     │
    └─────────────────────┴──────────────┴──────────┴────────────┘
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure the package is importable when run from repo root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, build_calibration_data, compute_perplexity, quantize


def _parse_bit_config(s: str) -> Tuple[float, List[int]]:
    """Parse '4.0:4,8,16' → (4.0, [4, 8, 16])."""
    target_str, options_str = s.split(":")
    return float(target_str), [int(b) for b in options_str.split(",")]


def run_benchmark(
    model_path: str,
    bit_configs: List[str],
    n_calib: int = 64,
    seq_len: int = 128,
    max_ppl_tokens: int = 4096,
    ppl_stride: int = 512,
    device: str = "auto",
) -> None:
    device_map = "auto" if (device == "auto" and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"\nModel: {model_path}")
    print(f"Device: {device_map}  |  dtype: {dtype}")
    print(f"Calibration: n={n_calib}, seq_len={seq_len}")
    print(f"PPL: max_tokens={max_ppl_tokens}, stride={ppl_stride}\n")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[dict] = []

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    print("Evaluating FP32 baseline …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    t0 = time.time()
    base_ppl = compute_perplexity(
        base_model, tokenizer,
        max_tokens=max_ppl_tokens, stride=ppl_stride
    )
    base_time = time.time() - t0
    rows.append({
        "label": "FP32 baseline",
        "actual_bits": 32.0,
        "ppl": base_ppl,
        "elapsed_s": base_time,
    })
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build calibration data once
    print(f"Building calibration data (n={n_calib}) …")
    calib_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    calib_data = build_calibration_data(tokenizer, n=n_calib, seq_len=seq_len)
    del calib_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── CSAQ configs ──────────────────────────────────────────────────────────
    for cfg_str in bit_configs:
        target_bits, bit_options = _parse_bit_config(cfg_str)
        label = f"CSAQ {target_bits}-bit [{','.join(map(str, bit_options))}]"
        print(f"\nQuantising: {label} …")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, torch_dtype=dtype
        )

        config = CSAQConfig(target_bits=target_bits, bit_options=bit_options)
        t0 = time.time()
        model, info = quantize(model, calib_data, config=config, verbose=True)
        quant_time = time.time() - t0

        print(f"  Actual bits: {info['actual_bits']:.3f}")
        print(f"  Quantisation time: {quant_time:.1f}s")
        print(f"  Evaluating PPL …")

        ppl = compute_perplexity(
            model, tokenizer,
            max_tokens=max_ppl_tokens, stride=ppl_stride
        )
        rows.append({
            "label": label,
            "actual_bits": info["actual_bits"],
            "ppl": ppl,
            "elapsed_s": quant_time,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print table ───────────────────────────────────────────────────────────
    baseline_ppl = rows[0]["ppl"]
    print("\n" + "─" * 72)
    print(f"{'Config':<30}  {'Bits':>8}  {'PPL':>8}  {'vs FP32':>10}  {'Time':>8}")
    print("─" * 72)
    for row in rows:
        if row["ppl"] == baseline_ppl:
            vs = "—"
        else:
            delta = (row["ppl"] / baseline_ppl - 1.0) * 100
            vs = f"+{delta:.1f}%"
        print(
            f"{row['label']:<30}  "
            f"{row['actual_bits']:>8.2f}  "
            f"{row['ppl']:>8.2f}  "
            f"{vs:>10}  "
            f"{row['elapsed_s']:>7.1f}s"
        )
    print("─" * 72 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ PPL benchmark")
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--bit_configs",
        nargs="+",
        default=["4.0:4,8,16", "3.0:4,8"],
        help="Format: 'target_bits:option1,option2' e.g. '4.0:4,8,16'",
    )
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--max_ppl_tokens", type=int, default=4096)
    p.add_argument("--ppl_stride", type=int, default=512)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    run_benchmark(
        model_path=args.model_path,
        bit_configs=args.bit_configs,
        n_calib=args.n_calib,
        seq_len=args.seq_len,
        max_ppl_tokens=args.max_ppl_tokens,
        ppl_stride=args.ppl_stride,
        device=args.device,
    )


if __name__ == "__main__":
    main()
