"""
examples/example.py — End-to-end CSAQ usage demonstration.

IMPORTANT: You must supply your own calibration texts.
            Pass sentences in your target language/domain.

Run:
    # Basic run with provided sample texts
    python examples/example.py

    # With your own calibration file (one sentence per line)
    python examples/example.py --calib_file my_konkani_corpus.txt

    # GPU + larger model
    python examples/example.py \\
        --model Qwen/Qwen1.5-1.8B \\
        --calib_file konkani_corpus.txt \\
        --device auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from csaq import CSAQConfig, build_calibration_data, quantize
from csaq.inference import CSAQInferenceEngine
from csaq.utils import compute_perplexity, export_csaq_model, generate_csaq_report

# ─────────────────────────────────────────────────────────────────────────────
# Sample texts — replace with your domain corpus.
# These are generic English sentences only for illustration.
# For Konkani quantisation, pass Konkani sentences via --calib_file.
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLE_TEXTS = [
    "The transformer architecture revolutionised natural language processing.",
    "Quantisation reduces model size while preserving inference quality.",
    "Causal salience measures how much each weight influences the model output.",
    "Low-resource languages deserve dedicated model compression strategies.",
    "Self-speculative decoding generates draft tokens and verifies them in one pass.",
    "The Jaccard similarity measures co-activation overlap between weight channels.",
    "Post-training quantisation does not require retraining the model from scratch.",
    "Calibration data should reflect the target language and domain of deployment.",
    "Bit-packing stores multiple low-precision values in a single byte.",
    "A 4-bit model uses 8× less memory than a full fp32 model.",
    "Asymmetric quantisation maps the weight range to [0, 2^bits - 1].",
    "Per-group scales give better accuracy than per-channel at the same bit-width.",
    "Language preservation through selective quantisation is an open research problem.",
    "The protection floor ensures the most salient weights stay at high precision.",
    "Speculative decoding acceptance rate measures how often draft tokens are kept.",
    "A high acceptance rate means the draft and verifier agree — faster generation.",
    "Domain-specific calibration protects the weights that matter for your language.",
    "The clique graph groups channels that fire together across calibration samples.",
    "Greedy bit assignment upgrades the highest-salience cliques first.",
    "Early stopping saves compute once the salience ranking stabilises.",
] * 4   # 80 samples total


def main(
    model_name: str = "Qwen/Qwen1.5-0.5B",
    calib_file: str | None = None,
    device: str = "cpu",
    calibration_domain: str = "example",
) -> None:
    print(f"\n{'='*60}")
    print(f"  CSAQ Example")
    print(f"  Model  : {model_name}")
    print(f"  Domain : {calibration_domain}")
    print(f"{'='*60}\n")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("Loading model …")
    device_map = "auto" if device == "auto" and torch.cuda.is_available() else device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Calibration texts ───────────────────────────────────────────────────
    if calib_file:
        with open(calib_file, encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(texts)} calibration lines from {calib_file}")
    else:
        texts = _SAMPLE_TEXTS
        print(f"Using {len(texts)} built-in sample texts (replace with your domain corpus).")

    calib = build_calibration_data(
        tokenizer,
        custom_texts=texts,
        seq_len=128,
        device=str(next(model.parameters()).device),
    )
    print(f"Calibration batches: {len(calib)}")

    # ── 3. Configure ──────────────────────────────────────────────────────────
    config = CSAQConfig(
        target_bits=4.0,
        bit_options=[4, 8, 16],
        clique_threshold=0.85,
        protection_floor=0.10,
        group_size=128,
    )

    # ── 4. Quantise ────────────────────────────────────────────────────────────
    model, info = quantize(
        model, calib, config=config, verbose=True,
        calibration_domain=calibration_domain,
    )

    print(f"\nActual avg bits : {info['actual_bits']:.3f}")
    print(f"Cliques found   : {info['cliques_count']}")
    print(f"Bit distribution: {info['tier_stats']}")

    # ── 5. Standard inference ──────────────────────────────────────────────────
    print("\n--- Standard inference ---")
    prompt = "The future of language preservation is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        next(model.parameters()).device
    )
    model.eval()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=30, do_sample=False)
    print(f"Prompt : {prompt}")
    print(f"Output : {tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)}")

    # ── 6. Speculative decoding ────────────────────────────────────────────────
    print("\n--- Self-Speculative Decoding ---")
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer, verbose=True)
    out_s, report = engine.generate(
        ids, speculative=True, lookahead=4, max_new_tokens=30, temperature=0.0,
    )
    print(f"Output : {tokenizer.decode(out_s[0][ids.shape[-1]:], skip_special_tokens=True)}")
    for k, v in report.summary().items():
        print(f"  {k:<28}: {v}")

    # ── 7. Perplexity on your eval texts ──────────────────────────────────────
    print("\n--- Perplexity on calibration texts (proxy eval) ---")
    ppl = compute_perplexity(
        model, tokenizer,
        eval_texts=texts[:20],
        max_tokens=512, stride=128, seq_len=64,
        device=str(next(model.parameters()).device),
    )
    print(f"PPL: {ppl:.3f}")
    info["ppl"] = ppl

    # ── 8. Export ─────────────────────────────────────────────────────────────
    save_dir = "./csaq_output"
    print(f"\n--- Exporting model to {save_dir} ---")
    generate_csaq_report(info, save_path=f"{save_dir}/CSAQ_Report.json")
    export_csaq_model(model, config, info["budget"], save_dir, info=info)
    print("\n✓ Example complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-0.5B")
    p.add_argument("--calib_file", default=None,
                   help="Path to plain-text calibration file (one sentence per line).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--calibration_domain", default="example")
    args = p.parse_args()
    main(
        model_name=args.model,
        calib_file=args.calib_file,
        device=args.device,
        calibration_domain=args.calibration_domain,
    )
