"""
example.py — End-to-end CSAQ usage on Qwen/Qwen1.5-0.5B (CPU-friendly).

Run:
    python example.py

For GPU + larger model:
    python example.py --model Qwen/Qwen1.5-1.8B --device auto
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from csaq import CSAQConfig, build_calibration_data, quantize
from csaq.inference import CSAQInferenceEngine
from csaq.utils import compute_perplexity, export_csaq_model, generate_csaq_report


def main(model_name: str = "Qwen/Qwen1.5-0.5B", device: str = "cpu") -> None:
    print(f"\n{'='*60}")
    print(f"  CSAQ Example — {model_name}")
    print(f"{'='*60}\n")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("Loading model and tokenizer…")
    device_map = "auto" if device == "auto" and torch.cuda.is_available() else device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Calibration data ────────────────────────────────────────────────────
    print("\nBuilding calibration data…")
    calib = build_calibration_data(tokenizer, n=32, seq_len=128)

    # ── 3. Configure ──────────────────────────────────────────────────────────
    config = CSAQConfig(
        target_bits=4.0,
        bit_options=[4, 8, 16],
        clique_threshold=0.85,
        protection_floor=0.10,
        group_size=128,
    )

    # ── 4. Quantise ────────────────────────────────────────────────────────────
    model, info = quantize(model, calib, config=config, verbose=True)

    print(f"\nActual average bits : {info['actual_bits']:.3f}")
    print(f"Cliques discovered  : {info['cliques_count']}")
    print(f"Bit distribution    : {info['tier_stats']}")

    # ── 5. Standard inference ──────────────────────────────────────────────────
    print("\n--- Standard inference ---")
    prompt = "The Eiffel Tower is located in"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=40, do_sample=False)
    generated = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Prompt : {prompt}")
    print(f"Output : {generated}")

    # ── 6. Speculative decoding ────────────────────────────────────────────────
    print("\n--- Self-Speculative Decoding ---")
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer, verbose=True)

    output_spec, report = engine.generate(
        input_ids,
        speculative=True,
        lookahead=4,
        max_new_tokens=40,
        temperature=0.0,
    )
    generated_spec = tokenizer.decode(
        output_spec[0][input_ids.shape[-1]:], skip_special_tokens=True
    )
    print(f"Output : {generated_spec}")
    print("\nSpeculative decoding report:")
    for k, v in report.summary().items():
        print(f"  {k:<28}: {v}")

    # ── 7. Perplexity (small slice for speed) ─────────────────────────────────
    print("\n--- Perplexity (512 tokens) ---")
    ppl = compute_perplexity(
        model, tokenizer,
        max_tokens=512, stride=128, seq_len=128,
        device=str(next(model.parameters()).device),
    )
    print(f"WikiText-2 PPL: {ppl:.2f}")

    # ── 8. Export ─────────────────────────────────────────────────────────────
    save_dir = "./csaq_output"
    print(f"\n--- Exporting model to {save_dir} ---")
    info["ppl"] = ppl
    generate_csaq_report(info, save_path=f"{save_dir}/CSAQ_Report.json")
    export_csaq_model(model, config, info["budget"], save_dir, info=info)

    print("\n✓ Example complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-0.5B")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    main(model_name=args.model, device=args.device)
