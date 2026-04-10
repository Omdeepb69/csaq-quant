"""
eval_suite.py — Production-Grade Validation Suite for CSAQ v0.3.3
Optimized for Google Colab and high-stability research deployments.
"""

import os
import time
import json
import torch
import gc
import warnings
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from csaq import (
    quantize, 
    CSAQConfig, 
    CSAQInferenceEngine, 
    build_calibration_data, 
    compute_perplexity
)

# Detect if running in Google Colab for appropriate reporting
try:
    from IPython.display import Markdown, display
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODELS_TO_TEST = ["Qwen/Qwen1.5-0.5B"] 
TARGET_BITS_LIST = [4.0, 6.0]
CALIB_SAMPLES = 16 
EVAL_TOKENS = 1024 
GEN_MAX_NEW_TOKENS = 64
SPEC_LOOKAHEAD = 4

# Path adjustments for Colab
REPORT_FILE = "/content/CSAQ_Final_Validation.md" if IS_COLAB else "./CSAQ_Final_Validation.md"

def log_to_report(text: str):
    with open(REPORT_FILE, "a") as f:
        f.write(text + "\n")

def garbage_collect():
    """Aggressive memory cleanup to prevent OOM in Colab."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def run_evaluation():
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
    
    log_to_report("# CSAQ Architectural Hardening Report (v0.3.3)")
    log_to_report(f"**Date:** {time.ctime()} ({'Google Colab' if IS_COLAB else 'Local'})")
    log_to_report("**Mode:** Accuracy Rescue (Protection Floor: 20%)")
    log_to_report("\n" + "="*80 + "\n")

    results_table = []

    for model_id in MODELS_TO_TEST:
        print(f"\n[SUITE] Evaluating Model: {model_id}")
        log_to_report(f"## Model: {model_id}")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"[SUITE] Loading FP16 Baseline...")
            model_fp16 = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print(f"[SUITE] Baseline Perplexity (WikiText-103)...")
            base_ppl = compute_perplexity(
                model_fp16, tokenizer, max_tokens=EVAL_TOKENS, device=device
            )
            log_to_report(f"- **Baseline PPL (FP16):** {base_ppl:.4f}")
            if torch.cuda.is_available():
                log_to_report(f"- **Peak VRAM (Baseline):** {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                torch.cuda.reset_peak_memory_stats()

            for t_bits in TARGET_BITS_LIST:
                garbage_collect()
                print(f"\n[SUITE] Applying CSAQ: {t_bits} bits")
                log_to_report(f"### Target: {t_bits} bits")
                
                config = CSAQConfig(
                    target_bits=t_bits,
                    bit_options=[1, 2, 4, 8, 16],
                    salience_alpha=0.03 # Optimized for deep quantization
                )
                
                calib_data = build_calibration_data(tokenizer, n=CALIB_SAMPLES, device=device, hard=True)
                
                q_model, info = quantize(model_fp16, calib_data, config, verbose=True)
                
                # Accuracy Evaluation
                q_ppl = compute_perplexity(
                    q_model, tokenizer, max_tokens=EVAL_TOKENS, device=device
                )
                ppl_delta = q_ppl - base_ppl
                log_to_report(f"- **Quantized PPL:** {q_ppl:.4f} (Δ {ppl_delta:.4f})")
                
                if torch.cuda.is_available():
                    log_to_report(f"- **Peak VRAM (Quantized):** {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                    torch.cuda.reset_peak_memory_stats()

                # SSD Benchmark
                engine = CSAQInferenceEngine(q_model, info["causal_map"], tokenizer)
                prompt = "The future of efficient AI hinges on"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                _, report_std = engine.generate(inputs.input_ids, max_new_tokens=GEN_MAX_NEW_TOKENS, speculative=False)
                _, report_spec = engine.generate(inputs.input_ids, max_new_tokens=GEN_MAX_NEW_TOKENS, speculative=True)
                
                speedup = report_std.inter_token_latency_ms / max(report_spec.inter_token_latency_ms, 1e-8)
                
                log_to_report(f"- **SSD Speedup:** {speedup:.2f}x")
                log_to_report(f"- **Acceptance Rate:** {report_spec.acceptance_rate*100:.2f}%")
                if report_spec.error_log:
                    log_to_report(f"- **Recovery Log:** {report_spec.error_log}")

                results_table.append({
                    "Model": model_id,
                    "Bits": t_bits,
                    "Speedup": round(speedup, 2),
                    "Acceptance": round(report_spec.acceptance_rate, 4)
                })
                
                # Periodic cleanup
                del engine
                garbage_collect()

            del model_fp16
            garbage_collect()

        except Exception as e:
            log_to_report(f"❌ **FAIL:** {str(e)}")
            print(f"Error: {e}")

    # Final Summary Table
    log_to_report("\n## Summary Table\n| Model | Bits | Speedup | Acceptance |\n|---|---|---|---|\n")
    for r in results_table:
        log_to_report(f"| {r['Model']} | {r['Bits']} | {r['Speedup']}x | {r['Acceptance']*100:.1f}% |")

    print(f"\n[SUITE] Hardened validation complete. Report: {REPORT_FILE}")
    if IS_COLAB:
        display(Markdown(open(REPORT_FILE).read()))

if __name__ == "__main__":
    run_evaluation()
