"""
csaq/__main__.py — Command-line interface for CSAQ quantisation.

Usage::

    # Basic 4-bit quantisation
    python -m csaq --model_path Qwen/Qwen1.5-0.5B --wbits 4.0 --save_path ./qwen-4bit

    # Aggressive 2-bit mix with hard calibration
    python -m csaq --model_path meta-llama/Llama-3-8B \\
                   --wbits 2.5 --options 2,4 --hard_calib --save_path ./llama-2.5bit

    # Evaluate perplexity after quantisation
    python -m csaq --model_path ./qwen-4bit --eval_ppl
"""

from __future__ import annotations

import argparse
import sys
import warnings

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csaq",
        description="CSAQ: Causal Salience-Aware Quantization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub model ID or local path.",
    )
    p.add_argument(
        "--save_path",
        required=True,
        help="Output directory for quantised model.",
    )

    # Quantisation options
    p.add_argument(
        "--wbits",
        type=float,
        default=4.0,
        help="Target average bits-per-weight (default: 4.0).",
    )
    p.add_argument(
        "--options",
        type=str,
        default="4,8,16",
        help="Comma-separated allowed bit widths (default: '4,8,16').",
    )
    p.add_argument(
        "--clique_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for clique grouping (default: 0.85).",
    )
    p.add_argument(
        "--group_size",
        type=int,
        default=-1,
        help="Per-group quantisation group size (-1 = per-channel, default).",
    )
    p.add_argument(
        "--protection_floor",
        type=float,
        default=0.10,
        help="Fraction of salient rows always kept at ≥8-bit (default: 0.10).",
    )

    # Calibration
    p.add_argument(
        "--n_calib",
        type=int,
        default=64,
        help="Number of calibration samples (default: 64).",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Calibration sequence length (default: 128).",
    )
    p.add_argument(
        "--hard_calib",
        action="store_true",
        help="Use hard calibration data (MATH + HumanEval + WikiText).",
    )

    # Evaluation
    p.add_argument(
        "--eval_ppl",
        action="store_true",
        help="Evaluate WikiText-2 perplexity after quantisation.",
    )
    p.add_argument(
        "--eval_baseline_ppl",
        action="store_true",
        help="Also evaluate baseline (fp32) perplexity for comparison.",
    )

    # Device
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc. (default: auto).",
    )

    # Misc
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    p.add_argument("--version", action="version", version="csaq 0.5.0")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    # ── Device resolution ────────────────────────────────────────────────────
    if args.device == "auto":
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        device_map = args.device
        torch_dtype = torch.float32

    # ── Imports ──────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from csaq import CSAQConfig, quantize
    from csaq.utils import (
        build_calibration_data,
        compute_perplexity,
        export_csaq_model,
        generate_csaq_report,
    )

    # ── Bit options ──────────────────────────────────────────────────────────
    bit_options = [int(b.strip()) for b in args.options.split(",")]

    # ── Load model ───────────────────────────────────────────────────────────
    if verbose:
        print(f"[CSAQ CLI] Loading {args.model_path} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        print(f"[CSAQ CLI] ERROR: Could not load model: {exc}", file=sys.stderr)
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Optional baseline PPL ────────────────────────────────────────────────
    if args.eval_baseline_ppl:
        if verbose:
            print("[CSAQ CLI] Evaluating baseline perplexity …")
        base_ppl = compute_perplexity(model, tokenizer)
        print(f"[CSAQ CLI] Baseline PPL: {base_ppl:.2f}")

    # ── Build calibration data ───────────────────────────────────────────────
    if verbose:
        print(f"[CSAQ CLI] Building calibration data (n={args.n_calib}) …")
    calib_data = build_calibration_data(
        tokenizer,
        n=args.n_calib,
        seq_len=args.seq_len,
        hard=args.hard_calib,
    )

    # ── Configure & run quantisation ─────────────────────────────────────────
    config = CSAQConfig(
        target_bits=args.wbits,
        bit_options=bit_options,
        clique_threshold=args.clique_threshold,
        group_size=args.group_size,
        protection_floor=args.protection_floor,
    )
    model, info = quantize(model, calib_data, config=config, verbose=verbose)

    # ── Optional post-quant PPL ──────────────────────────────────────────────
    if args.eval_ppl:
        if verbose:
            print("[CSAQ CLI] Evaluating post-quantisation perplexity …")
        quant_ppl = compute_perplexity(model, tokenizer)
        info["ppl"] = quant_ppl
        print(f"[CSAQ CLI] Quantised PPL: {quant_ppl:.2f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    report_path = f"{args.save_path}/CSAQ_Report.json"
    generate_csaq_report(info, save_path=report_path)
    export_csaq_model(model, config, info["budget"], args.save_path, info=info)

    if verbose:
        print(f"\n[CSAQ CLI] Done.  Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
