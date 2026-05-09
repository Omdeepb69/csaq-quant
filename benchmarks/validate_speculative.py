"""
benchmarks/validate_speculative.py — Speculative decoding validation.

Measures acceptance rate, tok/s, speedup, and p95 latency across
standard vs speculative generation at multiple lookahead values.

Usage::

    python benchmarks/validate_speculative.py \
        --model_path Qwen/Qwen1.5-0.5B \
        --calib_file calib.txt

    # With custom eval prompts and output path
    python benchmarks/validate_speculative.py \
        --model_path Qwen/Qwen1.5-0.5B \
        --calib_file calib.txt \
        --eval_prompts_file prompts.txt \
        --output_path results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, quantize
from csaq.inference import CSAQInferenceEngine
from csaq.utils import build_calibration_data

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "In the heart of the Amazon rainforest,",
    "La evolución de la inteligencia artificial ha cambiado",
    "量子コンピューティングの発展により、",
    "Die Geschichte der Mathematik beginnt mit",
]


def _measure_memory() -> float:
    """Return current memory usage in GB (VRAM if CUDA, else RSS)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _load_text_file(path: str, min_len: int = 10) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if len(l.strip()) >= min_len]


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    model_path: str,
    calib_file: str,
    eval_prompts_file: str | None,
    lookahead_values: List[int],
    n_tokens: int,
    n_runs: int,
    n_calib: int,
    seq_len: int,
    device: str,
    output_path: str | None,
) -> Dict[str, Any]:
    device_map = "auto" if device == "auto" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ── Load texts ───────────────────────────────────────────────────────
    calib_texts = _load_text_file(calib_file)[:n_calib]
    if not calib_texts:
        raise ValueError(f"No calibration texts found in {calib_file}")

    if eval_prompts_file:
        prompts = _load_text_file(eval_prompts_file, min_len=5)
        if not prompts:
            raise ValueError(f"No prompts found in {eval_prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS

    # ── Load & quantise ──────────────────────────────────────────────────
    print(f"\nModel      : {model_path}")
    print(f"Calib file : {calib_file}  ({len(calib_texts)} samples)")
    print(f"Prompts    : {len(prompts)} eval prompts")
    print(f"Device     : {device_map}  dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_before_load = _measure_memory()

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    mem_after_load = _measure_memory()

    calib_data = build_calibration_data(
        tokenizer, custom_texts=calib_texts, seq_len=seq_len
    )

    config = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16])
    model, info = quantize(model, calib_data, config=config, verbose=True)
    mem_after_quant = _measure_memory()

    actual_bits = info["actual_bits"]
    print(f"\nQuantised to {actual_bits:.2f} avg bits")
    print(f"Memory: load={mem_after_load:.2f}GB → quant={mem_after_quant:.2f}GB")

    # ── Build engine ─────────────────────────────────────────────────────
    engine = CSAQInferenceEngine(
        model, info["causal_map"], tokenizer=tokenizer, verbose=False
    )
    engine.warmup(n=2)

    # ── Encode prompts ───────────────────────────────────────────────────
    encoded_prompts = []
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors="pt")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        encoded_prompts.append(ids.to(next(model.parameters()).device))

    # ── Run benchmarks ───────────────────────────────────────────────────
    modes: List[Dict[str, Any]] = [
        {"label": "Standard (quantised)", "speculative": False, "lookahead": 0},
    ]
    for la in lookahead_values:
        modes.append({
            "label": f"Speculative la={la}",
            "speculative": True,
            "lookahead": la,
        })

    results: List[Dict[str, Any]] = []
    baseline_tps: float = 0.0

    for mode in modes:
        all_tps: List[float] = []
        all_accept: List[float] = []
        all_p95: List[float] = []

        for run_idx in range(n_runs):
            for prompt_ids in encoded_prompts:
                output, report = engine.generate(
                    prompt_ids,
                    speculative=mode["speculative"],
                    lookahead=mode.get("lookahead", 4),
                    max_new_tokens=n_tokens,
                    temperature=0.0,
                )
                tps = report.tokens_per_second
                all_tps.append(tps)
                all_accept.append(report.acceptance_rate)
                if report._token_times:
                    all_p95.append(report.p95_latency_ms)

        import numpy as np
        mean_tps = float(np.mean(all_tps)) if all_tps else 0.0
        mean_accept = float(np.mean(all_accept)) if all_accept else 0.0
        mean_p95 = float(np.mean(all_p95)) if all_p95 else float("nan")

        if mode["label"].startswith("Standard"):
            baseline_tps = mean_tps
            speedup = 1.0
        else:
            speedup = mean_tps / max(baseline_tps, 1e-6)

        row = {
            "label": mode["label"],
            "accept_rate": round(mean_accept * 100, 1),
            "tok_s": round(mean_tps, 1),
            "speedup": round(speedup, 2),
            "p95_ms": round(mean_p95, 2) if not (mean_p95 != mean_p95) else "nan",
        }
        results.append(row)
        print(f"  {row['label']:<22}  accept={row['accept_rate']:>5}%  "
              f"tok/s={row['tok_s']:>7}  speedup={row['speedup']:.2f}×")

    # ── Publication-ready table ──────────────────────────────────────────
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    header = f"Model: {model_path}  |  Bits: {actual_bits:.2f}  |  Device: {dev}"
    print(f"\n{header}")
    print("┌──────────────────────┬────────────┬──────────┬─────────┐")
    print("│ Mode                 │ Accept     │ tok/s    │ Speedup │")
    print("├──────────────────────┼────────────┼──────────┼─────────┤")
    for r in results:
        acc_str = "—" if r["label"].startswith("Standard") else f"{r['accept_rate']}%"
        print(f"│ {r['label']:<20} │ {acc_str:<10} │ {r['tok_s']:<8} │ {r['speedup']:.2f}×   │")
    print("└──────────────────────┴────────────┴──────────┴─────────┘")

    # ── Save JSON ────────────────────────────────────────────────────────
    output = {
        "model": model_path,
        "actual_bits": round(actual_bits, 3),
        "device": dev,
        "n_tokens": n_tokens,
        "n_runs": n_runs,
        "n_prompts": len(prompts),
        "memory_load_gb": round(mem_after_load, 3),
        "memory_quant_gb": round(mem_after_quant, 3),
        "results": results,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n[CSAQ] Results saved → {output_path}")

    return output


def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ speculative decoding validation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calib_file", required=True,
                   help="Plain-text calibration file (one sentence per line).")
    p.add_argument("--eval_prompts_file", default=None,
                   help="Optional plain-text file with eval prompts "
                        "(falls back to 5 built-in multilingual prompts).")
    p.add_argument("--lookahead", type=int, nargs="+", default=[4, 6, 8],
                   help="Lookahead values to test (default: 4 6 8)")
    p.add_argument("--n_tokens", type=int, default=128)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_path", default=None,
                   help="Path to save JSON results (optional).")
    args = p.parse_args()

    run_validation(
        model_path=args.model_path,
        calib_file=args.calib_file,
        eval_prompts_file=args.eval_prompts_file,
        lookahead_values=args.lookahead,
        n_tokens=args.n_tokens,
        n_runs=args.n_runs,
        n_calib=args.n_calib,
        seq_len=args.seq_len,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
