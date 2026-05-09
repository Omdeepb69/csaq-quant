"""
benchmarks/benchmark_ppl.py — PPL benchmark across bit configs.

Requires your own calibration and evaluation text files.
No dataset is loaded automatically.

Usage::

    # Konkani model benchmark
    python benchmarks/benchmark_ppl.py \\
        --model_path ai4bharat/indic-bert \\
        --calib_file konkani_train.txt \\
        --eval_file  konkani_test.txt \\
        --bit_configs "4.0:4,8,16" "3.0:4,8" \\
        --calibration_domain konkani

    # Standard WikiText-2 benchmark (to compare vs GPTQ/AWQ)
    # You must export and pass WikiText-2 yourself:
    #   python -c \"from datasets import load_dataset; ds=load_dataset('wikitext','wikitext-2-raw-v1');
    #              open('wt2_train.txt','w').writelines(t+'\\n' for t in ds['train']['text']);
    #              open('wt2_test.txt','w').writelines(t+'\\n' for t in ds['test']['text'])\"
    python benchmarks/benchmark_ppl.py \\
        --model_path Qwen/Qwen1.5-0.5B \\
        --calib_file wt2_train.txt \\
        --eval_file  wt2_test.txt \\
        --bit_configs "4.0:4,8,16" "3.0:4,8"
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, quantize
from csaq.utils import build_calibration_data, compute_perplexity


def _parse_bit_config(s: str) -> Tuple[float, List[int]]:
    target_str, options_str = s.split(":")
    return float(target_str), [int(b) for b in options_str.split(",")]


def _load_text_file(path: str, min_len: int = 80) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) >= min_len]
    return lines


def run_benchmark(
    model_path: str,
    calib_file: str,
    eval_file: str,
    bit_configs: List[str],
    n_calib: int = 64,
    seq_len: int = 128,
    max_eval_tokens: int = 4096,
    eval_stride: int = 512,
    device: str = "auto",
    calibration_domain: str = "user_provided",
) -> None:
    device_map = "auto" if device == "auto" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load texts
    calib_texts = _load_text_file(calib_file)[:n_calib]
    eval_texts = _load_text_file(eval_file, min_len=10)

    if not calib_texts:
        raise ValueError(f"No calibration texts found in {calib_file}")
    if not eval_texts:
        raise ValueError(f"No evaluation texts found in {eval_file}")

    print(f"\nModel          : {model_path}")
    print(f"Calib file     : {calib_file}  ({len(calib_texts)} samples)")
    print(f"Eval file      : {eval_file}   ({len(eval_texts)} samples)")
    print(f"Domain         : {calibration_domain}")
    print(f"Device         : {device_map}  dtype: {dtype}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[dict] = []

    # FP32 baseline
    print("Evaluating FP32 baseline …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )
    t0 = time.time()
    base_ppl = compute_perplexity(
        base_model, tokenizer, eval_texts,
        max_tokens=max_eval_tokens, stride=eval_stride, seq_len=seq_len,
    )
    rows.append({"label": "FP32 baseline", "actual_bits": 32.0, "ppl": base_ppl,
                 "elapsed_s": time.time() - t0})
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Pre-build calibration data once
    calib_data = build_calibration_data(
        tokenizer, custom_texts=calib_texts, seq_len=seq_len
    )

    # CSAQ configs
    for cfg_str in bit_configs:
        target_bits, bit_options = _parse_bit_config(cfg_str)
        label = f"CSAQ {target_bits}-bit [{','.join(map(str, bit_options))}]"
        print(f"\nQuantising: {label} …")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, torch_dtype=dtype
        )
        config = CSAQConfig(target_bits=target_bits, bit_options=bit_options)
        t0 = time.time()
        model, info = quantize(
            model, calib_data, config=config, verbose=True,
            calibration_domain=calibration_domain,
        )
        q_time = time.time() - t0
        ppl = compute_perplexity(
            model, tokenizer, eval_texts,
            max_tokens=max_eval_tokens, stride=eval_stride, seq_len=seq_len,
        )
        rows.append({
            "label": label, "actual_bits": info["actual_bits"],
            "ppl": ppl, "elapsed_s": q_time,
        })
        print(f"  Actual bits: {info['actual_bits']:.3f}  |  PPL: {ppl:.3f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Table
    baseline_ppl = rows[0]["ppl"]
    print("\n" + "─" * 68)
    print(f"{'Config':<30}  {'Bits':>6}  {'PPL':>8}  {'vs FP32':>9}  {'Time':>7}")
    print("─" * 68)
    for row in rows:
        vs = "—" if row["ppl"] == baseline_ppl else f"+{(row['ppl'] / baseline_ppl - 1) * 100:.1f}%"
        print(
            f"{row['label']:<30}  {row['actual_bits']:>6.2f}  "
            f"{row['ppl']:>8.3f}  {vs:>9}  {row['elapsed_s']:>6.1f}s"
        )
    print("─" * 68 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ PPL benchmark")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calib_file", required=True,
                   help="Plain-text calibration file (one sentence per line).")
    p.add_argument("--eval_file", required=True,
                   help="Plain-text evaluation file (one sentence per line).")
    p.add_argument("--bit_configs", nargs="+", default=["4.0:4,8,16"],
                   help="Format: 'target:opt1,opt2' e.g. '4.0:4,8,16'")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--max_eval_tokens", type=int, default=4096)
    p.add_argument("--eval_stride", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--calibration_domain", default="user_provided")
    args = p.parse_args()

    run_benchmark(
        model_path=args.model_path,
        calib_file=args.calib_file,
        eval_file=args.eval_file,
        bit_configs=args.bit_configs,
        n_calib=args.n_calib,
        seq_len=args.seq_len,
        max_eval_tokens=args.max_eval_tokens,
        eval_stride=args.eval_stride,
        device=args.device,
        calibration_domain=args.calibration_domain,
    )


if __name__ == "__main__":
    main()
