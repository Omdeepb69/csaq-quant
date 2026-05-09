"""
benchmarks/benchmark_speculative.py — Self-Speculative Decoding throughput benchmark.

Requires your own calibration file.  No dataset is loaded automatically.

Usage::

    python benchmarks/benchmark_speculative.py \\
        --model_path Qwen/Qwen1.5-0.5B \\
        --calib_file konkani_sentences.txt \\
        --eval_prompts "आमी कोंकणी उलयतात" "The future of AI is" \\
        --lookahead 4 6 8 \\
        --n_tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, quantize
from csaq.inference import CSAQInferenceEngine
from csaq.utils import build_calibration_data


def _load_text_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def run_benchmark(
    model_path: str,
    calib_file: str,
    eval_prompts: List[str],
    lookahead_values: List[int],
    n_tokens: int = 128,
    warmup: int = 3,
    n_calib: int = 32,
    target_bits: float = 4.0,
    calibration_domain: str = "user_provided",
) -> None:
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    calib_texts = _load_text_file(calib_file)[:n_calib]
    if not calib_texts:
        raise ValueError(f"No texts found in {calib_file}")

    print(f"\nModel        : {model_path}")
    print(f"Calib file   : {calib_file}  ({len(calib_texts)} samples)")
    print(f"Domain       : {calibration_domain}")
    print(f"Target bits  : {target_bits}")
    print(f"Max tokens   : {n_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    calib_data = build_calibration_data(
        tokenizer, custom_texts=calib_texts, seq_len=128
    )
    config = CSAQConfig(target_bits=target_bits, bit_options=[4, 8, 16])
    model, info = quantize(
        model, calib_data, config=config, verbose=False,
        calibration_domain=calibration_domain,
    )
    print(f"Actual bits  : {info['actual_bits']:.3f}  |  Cliques: {info['cliques_count']}")

    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer, verbose=False)
    device = next(model.parameters()).device

    if warmup > 0:
        print(f"\nWarming up ({warmup} runs) …")
        engine.warmup(n=warmup)

    all_results: List[dict] = []

    for prompt in eval_prompts:
        print(f"\n── Prompt: \"{prompt[:50]}{'…' if len(prompt)>50 else ''}\"")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Standard baseline
        times = []
        for _ in range(3):
            _, rep = engine.generate(ids.clone(), speculative=False, max_new_tokens=n_tokens)
            times.append(rep.total_wallclock_s)
        std_tps = n_tokens / (sum(times) / len(times))
        all_results.append({"prompt": prompt[:30], "mode": "Standard", "lookahead": "—",
                            "acceptance": "—", "tps": std_tps, "speedup": 1.0})

        # Speculative at each lookahead
        for la in lookahead_values:
            times, accepts = [], []
            for _ in range(3):
                _, rep = engine.generate(
                    ids.clone(), speculative=True, lookahead=la, max_new_tokens=n_tokens
                )
                times.append(rep.total_wallclock_s)
                accepts.append(rep.acceptance_rate)
            tps = n_tokens / (sum(times) / len(times))
            all_results.append({
                "prompt": prompt[:30], "mode": f"Speculative la={la}",
                "lookahead": la, "acceptance": f"{sum(accepts)/len(accepts):.2%}",
                "tps": tps, "speedup": tps / std_tps,
            })

    print("\n" + "─" * 72)
    print(f"{'Prompt':<32} {'Mode':<20} {'Accept':>8} {'tok/s':>7} {'Speedup':>8}")
    print("─" * 72)
    for r in all_results:
        print(
            f"{r['prompt']:<32} {r['mode']:<20} "
            f"{str(r['acceptance']):>8} {r['tps']:>7.1f} {r['speedup']:>7.2f}x"
        )
    print("─" * 72 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ speculative decoding benchmark")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calib_file", required=True,
                   help="Plain-text calibration file (one sentence per line).")
    p.add_argument("--eval_prompts", nargs="+",
                   default=["The future of language models is"],
                   help="Prompts to generate from during benchmark.")
    p.add_argument("--lookahead", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--n_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--n_calib", type=int, default=32)
    p.add_argument("--target_bits", type=float, default=4.0)
    p.add_argument("--calibration_domain", default="user_provided")
    args = p.parse_args()

    run_benchmark(
        model_path=args.model_path,
        calib_file=args.calib_file,
        eval_prompts=args.eval_prompts,
        lookahead_values=args.lookahead,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        n_calib=args.n_calib,
        target_bits=args.target_bits,
        calibration_domain=args.calibration_domain,
    )


if __name__ == "__main__":
    main()
