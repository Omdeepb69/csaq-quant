"""
csaq/__main__.py — CLI for CSAQ quantisation.

IMPORTANT: You must provide your own calibration texts.
There is no default dataset — calibration should reflect YOUR target domain.

Usage::

    # Konkani model — provide Konkani calibration sentences
    python -m csaq \\
        --model_path ai4bharat/indic-bert \\
        --calib_file konkani_sentences.txt \\
        --wbits 4.0 \\
        --save_path ./indic-bert-konkani-4bit \\
        --calibration_domain konkani

    # Code model — provide code calibration file
    python -m csaq \\
        --model_path Qwen/Qwen2.5-Coder-7B \\
        --calib_file my_code_samples.txt \\
        --wbits 4.0 \\
        --save_path ./qwen-coder-4bit \\
        --calibration_domain python_code

    # Provide texts directly on the command line
    python -m csaq \\
        --model_path gpt2 \\
        --calib_texts "First sentence." "Second sentence." "Third sentence." \\
        --wbits 4.0 \\
        --save_path ./gpt2-4bit
"""

from __future__ import annotations

import argparse
import sys

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csaq",
        description="CSAQ: Causal Salience-Aware Quantization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--model_path", required=True,
                   help="HuggingFace Hub model ID or local path.")
    p.add_argument("--save_path", required=True,
                   help="Output directory for the quantised model.")

    # ── Calibration (one of these is required) ────────────────────────────────
    calib_group = p.add_mutually_exclusive_group(required=True)
    calib_group.add_argument(
        "--calib_file",
        type=str,
        help=(
            "Path to a plain-text file with one calibration sentence per line "
            "in your target language/domain (e.g. Konkani sentences, code snippets). "
            "Minimum recommended: 64 lines."
        ),
    )
    calib_group.add_argument(
        "--calib_texts",
        nargs="+",
        help="Calibration sentences passed directly as CLI arguments.",
    )

    p.add_argument(
        "--calibration_domain",
        type=str,
        default="user_provided",
        help=(
            "Label for the calibration domain stored in the report "
            "(e.g. 'konkani', 'medical', 'code'). Default: 'user_provided'."
        ),
    )
    p.add_argument(
        "--n_calib",
        type=int,
        default=None,
        help="Max number of calibration samples to use. Default: all lines in file.",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for calibration tokenisation (default: 128).",
    )

    # ── Quantisation ──────────────────────────────────────────────────────────
    p.add_argument("--wbits", type=float, default=4.0,
                   help="Target average bits-per-weight (default: 4.0).")
    p.add_argument("--options", type=str, default="4,8,16",
                   help="Comma-separated allowed bit widths (default: '4,8,16').")
    p.add_argument("--clique_threshold", type=float, default=0.85,
                   help="Jaccard similarity threshold (default: 0.85).")
    p.add_argument("--group_size", type=int, default=-1,
                   help="Per-group scale granularity (-1 = per-channel).")
    p.add_argument("--protection_floor", type=float, default=0.10,
                   help="Fraction of salient rows kept at >= 8-bit (default: 0.10).")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help=(
            "Path to a plain-text evaluation file (one sentence per line) "
            "used to compute post-quantisation perplexity. "
            "Should be in the same language/domain as --calib_file."
        ),
    )
    p.add_argument(
        "--eval_baseline",
        action="store_true",
        help="Also compute baseline (fp32) perplexity for comparison.",
    )
    p.add_argument("--max_eval_tokens", type=int, default=4096,
                   help="Max tokens for PPL evaluation (default: 4096).")
    p.add_argument("--eval_stride", type=int, default=512,
                   help="Sliding-window stride for PPL (default: 512).")

    # ── Device / misc ─────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', etc. (default: auto).")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    p.add_argument("--version", action="version", version="csaq 0.5.1")

    return p


def _load_text_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def main() -> None:
    from typing import List  # noqa: F401

    parser = _build_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        device_map = args.device
        torch_dtype = torch.float32

    # ── Imports ───────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from csaq import CSAQConfig, quantize
    from csaq.utils import (
        build_calibration_data,
        compute_perplexity,
        export_csaq_model,
        generate_csaq_report,
    )

    # ── Load calibration texts ────────────────────────────────────────────────
    if args.calib_file:
        raw_texts = _load_text_file(args.calib_file)
        if verbose:
            print(f"[CSAQ CLI] Loaded {len(raw_texts)} calibration lines from {args.calib_file}")
    else:
        raw_texts = args.calib_texts
        if verbose:
            print(f"[CSAQ CLI] Using {len(raw_texts)} calibration texts from CLI")

    if not raw_texts:
        print("[CSAQ CLI] ERROR: No calibration texts found.", file=sys.stderr)
        sys.exit(1)

    # ── Load eval texts if requested ──────────────────────────────────────────
    eval_texts = None
    if args.eval_file:
        eval_texts = _load_text_file(args.eval_file)
        if verbose:
            print(f"[CSAQ CLI] Loaded {len(eval_texts)} evaluation lines from {args.eval_file}")

    # ── Bit options ───────────────────────────────────────────────────────────
    bit_options = [int(b.strip()) for b in args.options.split(",")]

    # ── Load model ────────────────────────────────────────────────────────────
    if verbose:
        print(f"[CSAQ CLI] Loading {args.model_path} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device_map, torch_dtype=torch_dtype
        )
    except Exception as exc:
        print(f"[CSAQ CLI] ERROR: Could not load model: {exc}", file=sys.stderr)
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Optional baseline PPL ─────────────────────────────────────────────────
    if args.eval_baseline and eval_texts:
        if verbose:
            print("[CSAQ CLI] Evaluating baseline (fp32) perplexity …")
        base_ppl = compute_perplexity(
            model, tokenizer, eval_texts,
            max_tokens=args.max_eval_tokens,
            stride=args.eval_stride,
            seq_len=args.seq_len,
        )
        print(f"[CSAQ CLI] Baseline PPL: {base_ppl:.2f}")

    # ── Build calibration data ────────────────────────────────────────────────
    if verbose:
        n_use = args.n_calib if args.n_calib else len(raw_texts)
        print(f"[CSAQ CLI] Building calibration data (n={n_use}, seq_len={args.seq_len}) …")
    calib_data = build_calibration_data(
        tokenizer,
        custom_texts=raw_texts,
        n=args.n_calib,
        seq_len=args.seq_len,
    )

    # ── Quantise ──────────────────────────────────────────────────────────────
    config = CSAQConfig(
        target_bits=args.wbits,
        bit_options=bit_options,
        clique_threshold=args.clique_threshold,
        group_size=args.group_size,
        protection_floor=args.protection_floor,
    )
    model, info = quantize(
        model, calib_data, config=config, verbose=verbose,
        calibration_domain=args.calibration_domain,
    )

    # ── Optional post-quant PPL ───────────────────────────────────────────────
    if eval_texts:
        if verbose:
            print("[CSAQ CLI] Evaluating post-quantisation perplexity …")
        quant_ppl = compute_perplexity(
            model, tokenizer, eval_texts,
            max_tokens=args.max_eval_tokens,
            stride=args.eval_stride,
            seq_len=args.seq_len,
        )
        info["ppl"] = quant_ppl
        print(f"[CSAQ CLI] Quantised PPL: {quant_ppl:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    report_path = f"{args.save_path}/CSAQ_Report.json"
    generate_csaq_report(info, save_path=report_path)
    export_csaq_model(model, config, info["budget"], args.save_path, info=info)

    if verbose:
        print(f"\n[CSAQ CLI] Done. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
