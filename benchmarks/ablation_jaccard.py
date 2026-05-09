import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr
from csaq.config import CSAQConfig
from csaq.core import CausalProfiler, solve_clique_budget, apply_csaq
from csaq.utils import build_calibration_data, compute_perplexity

def main():
    parser = argparse.ArgumentParser(description="CSAQ Jaccard vs per-channel ablation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to text file with calibration sentences")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to text file with evaluation sentences")
    args = parser.parse_args()

    print(f"Loading texts...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        calib_texts = [line.strip() for line in f if line.strip()]
    with open(args.eval_file, "r", encoding="utf-8") as f:
        eval_texts = [line.strip() for line in f if line.strip()]

    print(f"Loading tokenizer {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building calibration data on {device}...")
    calib_data = build_calibration_data(tokenizer, calib_texts, n=64, seq_len=128, device=device)

    def run_mode(mode: str):
        print(f"\n--- Running Mode: {mode} ---")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model.to(device)

        config = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16], clique_mode=mode)
        
        print(f"Profiling (clique_mode={mode})...")
        profiler = CausalProfiler(model, config)
        salience, cliques = profiler.profile(calib_data, verbose=False)
        
        mean_rho = float('nan')
        if mode == "jaccard":
            rho_values = []
            for name, layer_cliques in cliques.items():
                s_ten = salience[name].cpu().numpy()
                for c in layer_cliques:
                    if len(c) > 1:
                        rows = [s_ten[i] for i in c]
                        for i in range(len(rows)):
                            for j in range(i + 1, len(rows)):
                                rho, _ = spearmanr(rows[i], rows[j])
                                if not np.isnan(rho):
                                    rho_values.append(rho)
            if rho_values:
                mean_rho = float(np.mean(rho_values))
                
        budget, tier_stats, actual_bits = solve_clique_budget(salience, cliques, config)
        print("Applying quantization...")
        _ = apply_csaq(model, budget, salience, config, verbose=False)
        cliques_count = sum(len(v) for v in cliques.values())
        
        print("Computing perplexity...")
        ppl = compute_perplexity(model, tokenizer, eval_texts, device=device)
        
        return cliques_count, actual_bits, ppl, mean_rho

    jac_cliques, jac_bits, jac_ppl, jac_rho = run_mode("jaccard")
    pc_cliques, pc_bits, pc_ppl, pc_rho = run_mode("per_channel")

    diff_pct = ((jac_ppl - pc_ppl) / pc_ppl) * 100.0

    print("\n" + "="*80)
    print("Ablation Results: CSAQ Jaccard vs. per-channel")
    print("="*80)
    print(f"{'Config':<20} | {'Cliques':<10} | {'Avg bits':<10} | {'PPL':<10} | {'vs per-channel'}")
    print("-" * 75)
    print(f"{'CSAQ Jaccard':<20} | {jac_cliques:<10} | {jac_bits:<10.2f} | {jac_ppl:<10.2f} | {diff_pct:+.1f}%")
    print(f"{'CSAQ per-channel':<20} | {pc_cliques:<10} | {pc_bits:<10.2f} | {pc_ppl:<10.2f} | baseline")
    print("="*80)
    print(f"Intra-clique Spearman ρ (Jaccard mode): {jac_rho:.4f}")

if __name__ == "__main__":
    main()
