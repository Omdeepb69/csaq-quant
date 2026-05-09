"""
benchmarks/benchmark_speculative.py — Self-Speculative Decoding throughput benchmark.

Measures tokens/second for:
  - Standard autoregressive (quantised weights)
  - Self-speculative decoding (CSAQ draft+verify)

Usage::

    python benchmarks/benchmark_speculative.py \\
        --model_path Qwen/Qwen1.5-0.5B \\
        --lookahead 4 6 8 \\
        --n_tokens 128 \\
        --warmup 3
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, build_calibration_data, quantize
from csaq.inference import CSAQInferenceEngine


def benchmark_engine(
    model_path: str,
    lookahead_values: List[int],
    n_tokens: int = 128,
    warmup: int = 3,
    n_calib: int = 32,
    target_bits: float = 4.0,
) -> None:
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"\nModel: {model_path}")
    print(f"Quantisation: {target_bits}-bit  |  Max new tokens: {n_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantise
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    calib = build_calibration_data(tokenizer, n=n_calib, seq_len=128)
    config = CSAQConfig(target_bits=target_bits, bit_options=[4, 8, 16])
    model, info = quantize(model, calib, config=config, verbose=False)
    print(f"Actual bits: {info['actual_bits']:.3f}  |  Cliques: {info['cliques_count']}\n")

    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer, verbose=False)

    # Prompt
    prompt = "The theory of relativity states that"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    results: list[dict] = []

    # ── Warmup ────────────────────────────────────────────────────────────────
    if warmup > 0:
        print(f"Warming up ({warmup} runs) …")
        engine.warmup(n=warmup)

    # ── Standard baseline ─────────────────────────────────────────────────────
    print("Benchmarking standard generation …")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _, rep = engine.generate(input_ids.clone(), speculative=False, max_new_tokens=n_tokens)
        times.append(rep.total_wallclock_s)
    avg_std = sum(times) / len(times)
    std_tps = n_tokens / avg_std
    results.append({
        "label": "Standard (quantised)",
        "lookahead": "—",
        "acceptance": "—",
        "tps": std_tps,
        "speedup": 1.0,
    })

    # ── Speculative at each lookahead ─────────────────────────────────────────
    for la in lookahead_values:
        print(f"Benchmarking speculative (lookahead={la}) …")
        times = []
        acceptances = []
        for _ in range(3):
            _, rep = engine.generate(
                input_ids.clone(),
                speculative=True,
                lookahead=la,
                max_new_tokens=n_tokens,
            )
            times.append(rep.total_wallclock_s)
            acceptances.append(rep.acceptance_rate)
        avg_t = sum(times) / len(times)
        avg_acc = sum(acceptances) / len(acceptances)
        tps = n_tokens / avg_t
        results.append({
            "label": f"Speculative (la={la})",
            "lookahead": la,
            "acceptance": f"{avg_acc:.2%}",
            "tps": tps,
            "speedup": tps / std_tps,
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print(f"{'Mode':<26}  {'Accept':>8}  {'tok/s':>8}  {'Speedup':>9}")
    print("─" * 68)
    for r in results:
        print(
            f"{r['label']:<26}  "
            f"{str(r['acceptance']):>8}  "
            f"{r['tps']:>8.1f}  "
            f"{r['speedup']:>8.2f}x"
        )
    print("─" * 68 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ speculative decoding benchmark")
    p.add_argument("--model_path", required=True)
    p.add_argument("--lookahead", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--n_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--n_calib", type=int, default=32)
    p.add_argument("--target_bits", type=float, default=4.0)
    args = p.parse_args()

    benchmark_engine(
        model_path=args.model_path,
        lookahead_values=args.lookahead,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        n_calib=args.n_calib,
        target_bits=args.target_bits,
    )


if __name__ == "__main__":
    main()
