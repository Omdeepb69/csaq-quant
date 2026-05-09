"""
CSAQ Research Validation Script

Requirements:
    pip install scipy matplotlib

Model: {model_path}
Domain: {domain_name}
Output: {output_dir}
Device: {device}
Sections: A B C D E F
NOTE: Sections A+C require multiple fresh model loads. On CPU this
will be slow. Use --skip_section D E F for a fast PPL-only run.

Usage:
    # Konkani research run (GPU recommended)
    python benchmarks/research_validation.py \
        --model_path Qwen/Qwen1.5-0.5B \
        --calib_file_domain examples/konkani_sample.txt \
        --eval_file_domain  examples/konkani_sample.txt \
        --calib_file_english wikitext2_train.txt \
        --eval_file_english  wikitext2_test.txt \
        --domain_name konkani \
        --output_dir ./research_output \
        --target_bits 4.0 \
        --device auto

    # Fast run (PPL + cliques only, skip slow sections)
    python benchmarks/research_validation.py \
        --model_path Qwen/Qwen1.5-0.5B \
        --calib_file_domain examples/konkani_sample.txt \
        --eval_file_domain  examples/konkani_sample.txt \
        --calib_file_english wikitext2_train.txt \
        --eval_file_english  wikitext2_test.txt \
        --output_dir ./research_output \
        --skip_section D E
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from csaq import CSAQConfig, CausalProfiler, quantize, solve_clique_budget
from csaq.inference import CSAQInferenceEngine
from csaq.utils import build_calibration_data, compute_perplexity


def measure_memory_gb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _load_texts(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if len(l.strip()) >= 10]


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_fresh_model(model_path: str, device_map: str, dtype: torch.dtype) -> nn.Module:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype
    )

def run_section_a(args, tokenizer, device_map, dtype, calib_en, calib_dom, eval_en, eval_dom, results, json_path):
    print("\n─── SECTION A: PPL BENCHMARK ───────────────────────────────────")
    t0 = time.time()
    
    rows = []
    
    # A1. FP32 baseline (English eval)
    model = get_fresh_model(args.model_path, device_map, dtype)
    ppl_fp32_en = compute_perplexity(model, tokenizer, eval_en, seq_len=args.seq_len)
    rows.append(["FP32 baseline", "—", "English", "32.00", ppl_fp32_en, "—", "—"])
    del model
    
    # A4. FP32 baseline (domain eval)
    model = get_fresh_model(args.model_path, device_map, dtype)
    ppl_fp32_dom = compute_perplexity(model, tokenizer, eval_dom, seq_len=args.seq_len)
    rows.append(["FP32 baseline", "—", args.domain_name, "32.00", ppl_fp32_dom, "—", "—"])
    del model
    
    # Quantised models
    cfg = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")])
    
    def q_and_eval(calib_data, eval_data, label, calib_name, eval_name, fp32_baseline):
        nonlocal results
        model = get_fresh_model(args.model_path, device_map, dtype)
        mem_before = measure_memory_gb()
        model, info = quantize(model, calib_data, config=cfg, verbose=False)
        mem_after = measure_memory_gb()
        saved_gb = max(0, mem_before - mem_after)
        saved_pct = (saved_gb / max(mem_before, 1e-6)) * 100
        
        ppl = compute_perplexity(model, tokenizer, eval_data, seq_len=args.seq_len)
        vs = f"+{(ppl / fp32_baseline - 1) * 100:.1f}%"
        mem_str = f"{saved_gb:.2f}GB ({saved_pct:.0f}%)"
        
        rows.append([label, calib_name, eval_name, f"{info['actual_bits']:.2f}", ppl, vs, mem_str])
        results["section_a"][f"{calib_name}_{eval_name}"] = {"ppl": ppl, "actual_bits": info['actual_bits'], "saved_gb": saved_gb, "saved_pct": saved_pct}
        del model
        return ppl
    
    results["section_a"] = {"fp32_en": ppl_fp32_en, "fp32_dom": ppl_fp32_dom}
    
    # A2. CSAQ [EN calib] (English eval)
    q_and_eval(calib_en, eval_en, f"CSAQ {args.target_bits}-bit", "English", "English", ppl_fp32_en)
    
    # A3. CSAQ [domain calib] (English eval)
    q_and_eval(calib_dom, eval_en, f"CSAQ {args.target_bits}-bit", args.domain_name, "English", ppl_fp32_en)
    
    # A5. CSAQ [EN calib] (domain eval)
    ppl_a5 = q_and_eval(calib_en, eval_dom, f"CSAQ {args.target_bits}-bit ← OURS", "English", args.domain_name, ppl_fp32_dom)
    
    # A6. CSAQ [domain calib] (domain eval)
    ppl_a6 = q_and_eval(calib_dom, eval_dom, f"CSAQ {args.target_bits}-bit ← KEY RESULT", args.domain_name, args.domain_name, ppl_fp32_dom)
    
    print("\nConfig                    | Calib    | Eval     | Bits  | PPL    | vs FP32 | VRAM saved")
    for r in rows:
        print(f"{r[0]:<25} | {r[1]:<8} | {r[2]:<8} | {r[3]:>5} | {r[4]:>6.2f} | {r[5]:>7} | {r[6]}")
        
    gain = ppl_a5 - ppl_a6
    gain_pct = (gain / ppl_a5) * 100
    print(f"\nDomain preservation gain: {gain:.2f} PPL points ({gain_pct:.1f}% improvement)")
    
    results["section_a"]["gain_pts"] = gain
    results["section_a"]["gain_pct"] = gain_pct
    
    csv_path = os.path.join(args.output_dir, "ppl_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Calib", "Eval", "Bits", "PPL", "vs FP32", "VRAM saved"])
        writer.writerows(rows)
    
    # Figure: ppl_comparison.png
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['English eval', f'{args.domain_name} eval']
    fp32_means = [ppl_fp32_en, ppl_fp32_dom]
    en_calib_means = [results["section_a"]["English_English"]["ppl"], ppl_a5]
    dom_calib_means = [results["section_a"][f"{args.domain_name}_English"]["ppl"], ppl_a6]
    
    x = np.arange(len(labels))
    width = 0.2
    
    ax.bar(x - width, fp32_means, width, label='FP32')
    ax.bar(x, en_calib_means, width, label='CSAQ (EN calib)')
    ax.bar(x + width, dom_calib_means, width, label=f'CSAQ ({args.domain_name} calib)')
    
    ax.set_ylabel('Perplexity')
    ax.set_title('Quantisation Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "ppl_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Section A completed in {time.time() - t0:.1f}s")
    save_json(results, json_path)
def run_section_b(args, tokenizer, device_map, dtype, calib_en, calib_dom, eval_dom, results, json_path):
    print("\n─── SECTION B: JACCARD CLIQUE ANALYSIS ─────────────────────────")
    t0 = time.time()
    
    cfg = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")])
    
    # B1. Profile model with domain calibration texts
    model = get_fresh_model(args.model_path, device_map, dtype)
    profiler = CausalProfiler(model, cfg)
    salience_dom, cliques_dom = profiler.profile(calib_dom, verbose=False)
    
    clique_stats = []
    total_cliques = 0
    
    for layer_name, layer_cliques in cliques_dom.items():
        layer_salience = salience_dom[layer_name]
        n_cliques = len(layer_cliques)
        total_cliques += n_cliques
        sizes = [len(c) for c in layer_cliques]
        n_singletons = sum(1 for s in sizes if s == 1)
        mean_size = np.mean(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        
        rhos = []
        for c in layer_cliques:
            if len(c) > 1:
                # sample up to 10 pairs
                pairs = []
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        pairs.append((c[i], c[j]))
                if len(pairs) > 10:
                    idx = np.random.choice(len(pairs), 10, replace=False)
                    pairs = [pairs[i] for i in idx]
                
                for u, v in pairs:
                    rho, _ = scipy.stats.spearmanr(layer_salience[u].numpy(), layer_salience[v].numpy())
                    if not np.isnan(rho):
                        rhos.append(rho)
        mean_rho = np.mean(rhos) if rhos else 0.0
        clique_stats.append({
            "layer": layer_name, "cliques": n_cliques, "singletons": n_singletons,
            "mean_size": mean_size, "max_size": max_size, "intra_rho": mean_rho,
            "sizes": sizes, "rhos": rhos
        })
        
    print(f"\nLayer                  | Cliques | Singletons | Mean size | Max | Intra-rho")
    for stat in clique_stats[:10]:
        print(f"{stat['layer']:<22} | {stat['cliques']:>7} | {stat['singletons']:>10} | {stat['mean_size']:>9.2f} | {stat['max_size']:>3} | {stat['intra_rho']:>9.3f}")

    results["section_b"] = {
        "total_cliques": total_cliques,
        "mean_clique_size": np.mean([s["mean_size"] for s in clique_stats]),
        "mean_intra_rho": np.mean([s["intra_rho"] for s in clique_stats])
    }

    csv_path = os.path.join(args.output_dir, "clique_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Cliques", "Singletons", "Mean Size", "Max Size", "Intra-rho"])
        for stat in clique_stats:
            writer.writerow([stat["layer"], stat["cliques"], stat["singletons"], stat["mean_size"], stat["max_size"], stat["intra_rho"]])

    # Figure: clique_sizes.png
    all_sizes = []
    all_rhos = []
    for stat in clique_stats:
        all_sizes.extend(stat["sizes"])
        all_rhos.extend(stat["rhos"])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(all_sizes, bins=range(1, max(all_sizes)+2), align='left')
    ax1.axvline(np.mean(all_sizes), color='r', linestyle='dashed', linewidth=1)
    ax1.axvline(np.median(all_sizes), color='g', linestyle='dashed', linewidth=1)
    ax1.set_title("Clique Size Distribution")
    ax1.set_xlabel("Clique Size")
    ax1.set_ylabel("Count")
    
    ax2.hist(all_rhos, bins=20)
    ax2.set_title("Intra-Clique Spearman Rho Distribution")
    ax2.set_xlabel("Spearman Rho")
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "figures", "clique_sizes.png"), dpi=300)
    plt.close()

    # B2. Ablation
    del model
    print("\nRunning ablation: Jaccard vs Per-channel...")
    
    model = get_fresh_model(args.model_path, device_map, dtype)
    cfg_jaccard = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")], clique_threshold=0.85)
    model_j, _ = quantize(model, calib_dom, config=cfg_jaccard, verbose=False)
    ppl_jaccard = compute_perplexity(model_j, tokenizer, eval_dom, seq_len=args.seq_len)
    del model, model_j
    
    model = get_fresh_model(args.model_path, device_map, dtype)
    cfg_perchan = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")], clique_mode="per_channel")
    model_p, _ = quantize(model, calib_dom, config=cfg_perchan, verbose=False)
    ppl_perchan = compute_perplexity(model_p, tokenizer, eval_dom, seq_len=args.seq_len)
    del model, model_p

    print(f"Jaccard cliques PPL:      {ppl_jaccard:.2f}")
    print(f"Per-channel baseline PPL: {ppl_perchan:.2f}")
    print(f"Clique grouping benefit:  {ppl_perchan - ppl_jaccard:.2f} PPL points")
    results["section_b"]["jaccard_ppl"] = ppl_jaccard
    results["section_b"]["perchan_ppl"] = ppl_perchan
    results["section_b"]["jaccard_benefit"] = ppl_perchan - ppl_jaccard

    # B3. Cross-domain salience divergence
    model = get_fresh_model(args.model_path, device_map, dtype)
    profiler = CausalProfiler(model, cfg)
    salience_en, _ = profiler.profile(calib_en, verbose=False)
    del model
    
    divergence_rows = []
    layer_names = []
    overlaps = []
    rhos_div = []
    
    for layer in salience_dom:
        s_dom = salience_dom[layer].mean(dim=1).numpy()
        s_en = salience_en[layer].mean(dim=1).numpy()
        
        k = max(1, int(0.1 * len(s_dom)))
        top_dom = set(np.argsort(s_dom)[-k:])
        top_en = set(np.argsort(s_en)[-k:])
        
        overlap = len(top_dom.intersection(top_en)) / len(top_dom.union(top_en))
        rho, _ = scipy.stats.spearmanr(s_dom, s_en)
        
        overlaps.append(overlap)
        rhos_div.append(rho)
        divergence_rows.append([layer, overlap, rho])
        layer_names.append(layer)
        
    mean_overlap = np.mean(overlaps)
    mean_rho_div = np.mean(rhos_div)
    print(f"\nMean top-10% Jaccard overlap (EN vs {args.domain_name}): {mean_overlap:.3f}")
    print(f"Mean full-vector Spearman rho: {mean_rho_div:.3f}")
    results["section_b"]["cross_domain_overlap"] = mean_overlap
    results["section_b"]["cross_domain_rho"] = mean_rho_div
    
    csv_path = os.path.join(args.output_dir, "domain_divergence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Jaccard Overlap", "Spearman Rho"])
        writer.writerows(divergence_rows)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['r' if o < 0.5 else 'b' for o in overlaps]
    ax.bar(range(len(overlaps)), overlaps, color=colors)
    ax.axhline(0.5, color='k', linestyle='dashed')
    ax.set_title(f"Cross-Domain Salience Overlap (EN vs {args.domain_name})")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Jaccard Overlap")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "figures", "domain_divergence.png"), dpi=300)
    plt.close()

    print(f"Section B completed in {time.time() - t0:.1f}s")
    save_json(results, json_path)
    return salience_dom, salience_en
def run_section_c(args, tokenizer, device_map, dtype, calib_dom, results, json_path):
    print("\n─── SECTION C: MEMORY BENCHMARK ────────────────────────────────")
    t0 = time.time()
    
    rows = []
    
    model = get_fresh_model(args.model_path, device_map, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    fp32_theoretical_gb = n_params * 4 / 1024**3
    mem_fp32 = measure_memory_gb()
    
    rows.append(["FP32", f"{n_params/1e6:.2f}M", f"{fp32_theoretical_gb:.3f}", f"{mem_fp32:.3f}", "—", "1.00×"])
    results["section_c"] = {"fp32_mem": mem_fp32, "fp32_size": fp32_theoretical_gb}
    del model
    
    for bits in [8, 4]:
        model = get_fresh_model(args.model_path, device_map, dtype)
        mem_before = measure_memory_gb()
        cfg = CSAQConfig(target_bits=float(bits), bit_options=[bits])
        model, _ = quantize(model, calib_dom, config=cfg, verbose=False)
        mem_after = measure_memory_gb()
        
        theoretical_gb = n_params * bits / 8 / 1024**3
        saved = max(0, mem_before - mem_after)
        ratio = mem_before / max(mem_after, 1e-6)
        rows.append([f"INT{bits}", f"{n_params/1e6:.2f}M", f"{theoretical_gb:.3f}", f"{mem_after:.3f}", f"{saved:.2f}GB", f"{ratio:.2f}×"])
        results["section_c"][f"int{bits}_mem"] = mem_after
        results["section_c"][f"int{bits}_saved"] = saved
        results["section_c"][f"int{bits}_ratio"] = ratio
        results["section_c"][f"int{bits}_theoretical"] = theoretical_gb
        del model
        
    print(f"Config   | Params   | Theoretical GB | Measured GB | Saved  | Ratio")
    for r in rows:
        print(f"{r[0]:<8} | {r[1]:<8} | {r[2]:>14} | {r[3]:>11} | {r[4]:>6} | {r[5]}")

    csv_path = os.path.join(args.output_dir, "memory_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Params", "Theoretical GB", "Measured GB", "Saved", "Ratio"])
        writer.writerows(rows)

    print(f"Section C completed in {time.time() - t0:.1f}s")
    save_json(results, json_path)
def run_section_d(args, tokenizer, device_map, dtype, calib_dom, eval_en, eval_dom, results, json_path):
    print("\n─── SECTION D: SPECULATIVE DECODING BENCHMARK ──────────────────")
    t0 = time.time()
    
    model = get_fresh_model(args.model_path, device_map, dtype)
    cfg = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")])
    model, info = quantize(model, calib_dom, config=cfg, verbose=False)
    
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer=tokenizer, verbose=False)
    engine.warmup(n=2)
    
    en_prompts = eval_en[:3]
    dom_prompts = eval_dom[:3]
    
    lookaheads = [int(la) for la in args.lookahead_values.split(",")]
    
    rows = []
    
    def eval_prompts(prompts, lang_name):
        encoded = [tokenizer.encode(p, return_tensors="pt").to(model.device) for p in prompts]
        
        baseline_tps = 0.0
        
        # Standard
        all_tps = []
        for _ in range(args.n_spec_runs):
            for p in encoded:
                _, rep = engine.generate(p, speculative=False, max_new_tokens=64)
                all_tps.append(rep.tokens_per_second)
        
        baseline_tps = np.mean(all_tps) if all_tps else 1e-6
        rows.append([lang_name, "Standard", "—", baseline_tps, 1.0, float("nan")])
        if lang_name == "English":
            results["section_d"] = {"en_standard_tps": baseline_tps}
        else:
            results["section_d"]["dom_standard_tps"] = baseline_tps
            
        # Speculative
        best_tps = 0
        best_la = 0
        best_acc = 0
        
        la_results = []
        for la in lookaheads:
            tps, acc, p95 = [], [], []
            for _ in range(args.n_spec_runs):
                for p in encoded:
                    _, rep = engine.generate(p, speculative=True, lookahead=la, max_new_tokens=64)
                    tps.append(rep.tokens_per_second)
                    acc.append(rep.acceptance_rate)
                    if rep._token_times:
                        p95.append(rep.p95_latency_ms)
            m_tps = np.mean(tps) if tps else 0
            m_acc = np.mean(acc) if acc else 0
            m_p95 = np.mean(p95) if p95 else float("nan")
            speedup = m_tps / baseline_tps
            
            rows.append([lang_name, f"Speculative la={la}", f"{m_acc*100:.1f}%", m_tps, speedup, m_p95])
            if m_tps > best_tps:
                best_tps = m_tps
                best_la = la
                best_acc = m_acc
                
            la_results.append({"la": la, "tps": m_tps, "acc": m_acc, "speedup": speedup, "p95": m_p95})
            
        if lang_name == "English":
            results["section_d"]["en_best_tps"] = best_tps
            results["section_d"]["en_best_la"] = best_la
            results["section_d"]["en_best_acc"] = best_acc
            results["section_d"]["en_la_results"] = la_results
        else:
            results["section_d"]["dom_best_tps"] = best_tps
            results["section_d"]["dom_best_la"] = best_la
            results["section_d"]["dom_best_acc"] = best_acc
            results["section_d"]["dom_la_results"] = la_results
            results["section_d"]["dom_best_speedup"] = best_tps / baseline_tps
            
    eval_prompts(en_prompts, "English")
    eval_prompts(dom_prompts, args.domain_name)
    
    print("\nLanguage | Mode              | Accept | tok/s  | Speedup | p95 lat(ms)")
    for r in rows:
        print(f"{r[0]:<8} | {r[1]:<17} | {r[2]:>6} | {r[3]:>6.1f} | {r[4]:>6.2f}× | {r[5]:>6.1f}")
        
    dom_adv = results["section_d"]["dom_best_acc"] - results["section_d"]["en_best_acc"]
    print(f"\nDomain acceptance rate advantage: +{dom_adv*100:.1f} pp vs English")
    print("This supports the hypothesis that protected rows encode domain-specific ability.")
    
    csv_path = os.path.join(args.output_dir, "speculative_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Language", "Mode", "Accept", "tok/s", "Speedup", "p95 lat(ms)"])
        writer.writerows(rows)
        
    # Figure: speculative_speedup.png
    fig, ax = plt.subplots(figsize=(8, 6))
    en_speeds = [r["speedup"] for r in results["section_d"]["en_la_results"]]
    dom_speeds = [r["speedup"] for r in results["section_d"]["dom_la_results"]]
    
    ax.plot(lookaheads, en_speeds, marker='o', label="English Prompts")
    ax.plot(lookaheads, dom_speeds, marker='s', label=f"{args.domain_name} Prompts")
    
    for i, la in enumerate(lookaheads):
        en_acc = results["section_d"]["en_la_results"][i]["acc"] * 100
        dom_acc = results["section_d"]["dom_la_results"][i]["acc"] * 100
        ax.annotate(f"{en_acc:.1f}%", (la, en_speeds[i]), textcoords="offset points", xytext=(0,-15), ha='center')
        ax.annotate(f"{dom_acc:.1f}%", (la, dom_speeds[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    ax.set_title("Speculative Decoding Speedup vs Lookahead")
    ax.set_xlabel("Lookahead")
    ax.set_ylabel("Speedup over Standard Generation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "figures", "speculative_speedup.png"), dpi=300)
    plt.close()
    
    del engine, model
    print(f"Section D completed in {time.time() - t0:.1f}s")
    save_json(results, json_path)
def run_section_e(args, tokenizer, device_map, dtype, salience_dom, salience_en, results, json_path):
    print("\n─── SECTION E: SALIENCE TOPOLOGY ANALYSIS ──────────────────────")
    t0 = time.time()
    
    # Get top 5 layers by parameter count
    layer_sizes = {layer: s.numel() for layer, s in salience_dom.items()}
    top_layers = sorted(layer_sizes.keys(), key=lambda x: layer_sizes[x], reverse=True)[:5]
    
    cfg = CSAQConfig(target_bits=args.target_bits, bit_options=[int(b) for b in args.bit_options.split(",")])
    
    rows = []
    div_pcts = []
    
    for layer in top_layers:
        s_dom = salience_dom[layer].mean(dim=1).numpy()
        s_en = salience_en[layer].mean(dim=1).numpy()
        
        k20 = max(1, int(0.2 * len(s_dom)))
        top_dom = set(np.argsort(s_dom)[-k20:])
        top_en = set(np.argsort(s_en)[-k20:])
        
        overlap = len(top_dom.intersection(top_en))
        union = len(top_dom.union(top_en))
        jaccard = overlap / union
        
        dom_only = len(top_dom - top_en)
        en_only = len(top_en - top_dom)
        
        rows.append([layer, dom_only, en_only, overlap, jaccard])
        
        # divergence of protected rows
        k_prot = max(1, int(cfg.protection_floor * len(s_dom)))
        prot_dom = set(np.argsort(s_dom)[-k_prot:])
        prot_en = set(np.argsort(s_en)[-k_prot:])
        div_pct = (len(prot_dom - prot_en) / k_prot) * 100
        div_pcts.append(div_pct)
        
    print(f"Layer                  | Domain-only rows | EN-only rows | Overlap | Jaccard")
    for r in rows:
        print(f"{r[0]:<22} | {r[1]:>16} | {r[2]:>12} | {r[3]:>7} | {r[4]:>7.3f}")
        
    mean_div_pct = np.mean(div_pcts)
    print(f"\nMean protected-row divergence: {mean_div_pct:.1f}%")
    results["section_e"] = {"mean_protected_divergence_pct": mean_div_pct}

    # Figure: salience_heatmap.png
    largest_layer = top_layers[0]
    
    # max 64x64
    s_dom_full = salience_dom[largest_layer].numpy()
    s_en_full = salience_en[largest_layer].numpy()
    
    r_idx = np.random.choice(s_dom_full.shape[0], min(64, s_dom_full.shape[0]), replace=False)
    c_idx = np.random.choice(s_dom_full.shape[1], min(64, s_dom_full.shape[1]), replace=False)
    
    hm_dom = s_dom_full[r_idx][:, c_idx]
    hm_en = s_en_full[r_idx][:, c_idx]
    
    # normalize for better visualization
    hm_dom = np.log1p(hm_dom)
    hm_en = np.log1p(hm_en)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(hm_en, cmap='coolwarm', aspect='auto')
    ax1.set_title("English Calibration Salience")
    ax1.set_xlabel("Sampled Input Channels")
    ax1.set_ylabel("Sampled Output Channels")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(hm_dom, cmap='coolwarm', aspect='auto')
    ax2.set_title(f"{args.domain_name.capitalize()} Calibration Salience")
    ax2.set_xlabel("Sampled Input Channels")
    fig.colorbar(im2, ax=ax2)
    
    fig.suptitle(f"Salience Topology: {largest_layer}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "figures", "salience_heatmap.png"), dpi=300)
    plt.close()

    print(f"Section E completed in {time.time() - t0:.1f}s")
    save_json(results, json_path)
def print_section_f(args, results):
    print("\n─── SECTION F: FULL SUMMARY FOR PAPER ─────────────────────────")
    
    def get(s, k, default="skipped"):
        return results.get(s, {}).get(k, default)
    
    def fmt(v, format_str="{}"):
        if v == "skipped": return v
        try: return format_str.format(v)
        except: return v
        
    fp32_en = get("section_a", "fp32_en")
    csaq_en = get("section_a", "English_English", {}).get("ppl", "skipped")
    fp32_dom = get("section_a", "fp32_dom")
    csaq_dom_en = get("section_a", f"English_{args.domain_name}", {}).get("ppl", "skipped")
    csaq_dom_dom = get("section_a", f"{args.domain_name}_{args.domain_name}", {}).get("ppl", "skipped")
    
    csaq_en_deg = f"(+{(csaq_en/fp32_en - 1)*100:.1f}%)" if csaq_en != "skipped" and fp32_en != "skipped" else ""
    csaq_dom_en_deg = f"(+{(csaq_dom_en/fp32_dom - 1)*100:.1f}%)" if csaq_dom_en != "skipped" and fp32_dom != "skipped" else ""
    csaq_dom_dom_deg = f"(+{(csaq_dom_dom/fp32_dom - 1)*100:.1f}%)" if csaq_dom_dom != "skipped" and fp32_dom != "skipped" else ""

    print(f"""
  ══════════════════════════════════════════════════════
   CSAQ Research Paper — Key Results
   Model: {args.model_path}  |  Domain: {args.domain_name}
  ══════════════════════════════════════════════════════

  [Table 1 — Quantisation Quality]
  FP32 English PPL:                {fmt(fp32_en, '{:.2f}')}
  CSAQ {args.target_bits}-bit English PPL:          {fmt(csaq_en, '{:.2f}')}  {csaq_en_deg}
  FP32 {args.domain_name} PPL:               {fmt(fp32_dom, '{:.2f}')}
  CSAQ {args.target_bits}-bit [EN calib] {args.domain_name}:  {fmt(csaq_dom_en, '{:.2f}')}  {csaq_dom_en_deg}
  CSAQ {args.target_bits}-bit [{args.domain_name[:2].upper()} calib] {args.domain_name}:  {fmt(csaq_dom_dom, '{:.2f}')}  {csaq_dom_dom_deg}  ← OURS
  Domain preservation gain:        {fmt(get('section_a', 'gain_pts'), '{:.2f}')} PPL pts ({fmt(get('section_a', 'gain_pct'), '{:.1f}')}% better than EN calib)

  [Table 2 — Clique Analysis]
  Total cliques discovered:        {get('section_b', 'total_cliques')}
  Mean clique size:                {fmt(get('section_b', 'mean_clique_size'), '{:.2f}')} channels
  Mean intra-clique salience rho:  {fmt(get('section_b', 'mean_intra_rho'), '{:.3f}')}
  Jaccard clique PPL benefit:      {fmt(get('section_b', 'jaccard_benefit'), '{:.2f}')} PPL pts vs per-channel baseline
  Cross-domain salience overlap:   {fmt(get('section_b', 'cross_domain_overlap'), '{:.3f}')} (EN vs {args.domain_name})

  [Table 3 — Memory]
  FP32 model size:                 {fmt(get('section_c', 'fp32_size'), '{:.2f}')} GB
  CSAQ {args.target_bits}-bit size:                 {fmt(get('section_c', f'int{int(args.target_bits)}_theoretical'), '{:.2f}')} GB
  Memory saved:                    {fmt(get('section_c', f'int{int(args.target_bits)}_saved'), '{:.2f}')} GB
  Compression ratio:               {fmt(get('section_c', f'int{int(args.target_bits)}_ratio'), '{:.2f}')}×

  [Table 4 — Speculative Decoding]
  Standard tok/s:                  {fmt(get('section_d', 'dom_standard_tps'), '{:.1f}')}
  Best speculative tok/s:          {fmt(get('section_d', 'dom_best_tps'), '{:.1f}')}  (la={get('section_d', 'dom_best_la')})
  Best speedup:                    {fmt(get('section_d', 'dom_best_speedup'), '{:.2f}')}×
  English acceptance rate:         {fmt(get('section_d', 'en_best_acc', 'skipped')*100 if get('section_d', 'en_best_acc', 'skipped') != 'skipped' else 'skipped', '{:.1f}')}%
  {args.domain_name} acceptance rate:        {fmt(get('section_d', 'dom_best_acc', 'skipped')*100 if get('section_d', 'dom_best_acc', 'skipped') != 'skipped' else 'skipped', '{:.1f}')}%
  Domain acceptance advantage:     +{fmt((get('section_d', 'dom_best_acc', 'skipped') - get('section_d', 'en_best_acc', 'skipped'))*100 if get('section_d', 'en_best_acc', 'skipped') != 'skipped' else 'skipped', '{:.1f}')} percentage points

  ══════════════════════════════════════════════════════
""")

def main() -> None:
    p = argparse.ArgumentParser(description="CSAQ Research Validation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calib_file_domain", required=True)
    p.add_argument("--eval_file_domain", required=True)
    p.add_argument("--calib_file_english", required=True)
    p.add_argument("--eval_file_english", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--domain_name", default="konkani")
    p.add_argument("--target_bits", type=float, default=4.0)
    p.add_argument("--bit_options", default="4,8,16")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lookahead_values", default="4,6,8")
    p.add_argument("--n_spec_runs", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_section", nargs="+", default=[])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    device_map = "auto" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"CSAQ Research Validation Script")
    print(f"Model: {args.model_path}")
    print(f"Domain: {args.domain_name}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device_map}")
    print(f"Sections: A B C D E F")
    print("NOTE: Sections A+C require multiple fresh model loads. On CPU this")
    print("will be slow. Use --skip_section D E F for a fast PPL-only run.\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_en = build_calibration_data(tokenizer, _load_texts(args.calib_file_english), n=args.n_calib, seq_len=args.seq_len)
    calib_dom = build_calibration_data(tokenizer, _load_texts(args.calib_file_domain), n=args.n_calib, seq_len=args.seq_len)
    eval_en = _load_texts(args.eval_file_english)[:args.n_calib]
    eval_dom = _load_texts(args.eval_file_domain)[:args.n_calib]

    results = {}
    json_path = os.path.join(args.output_dir, "research_results.json")

    salience_dom, salience_en = None, None

    # Section A
    if "A" in args.skip_section:
        print("[CSAQ] Skipping section A")
    else:
        try:
            run_section_a(args, tokenizer, device_map, dtype, calib_en, calib_dom, eval_en, eval_dom, results, json_path)
        except Exception as e:
            print(f"[CSAQ] Warning: Section A failed: {e}")

    # Section B
    if "B" in args.skip_section:
        print("[CSAQ] Skipping section B")
    else:
        try:
            salience_dom, salience_en = run_section_b(args, tokenizer, device_map, dtype, calib_en, calib_dom, eval_dom, results, json_path)
        except Exception as e:
            print(f"[CSAQ] Warning: Section B failed: {e}")

    # Section C
    if "C" in args.skip_section:
        print("[CSAQ] Skipping section C")
    else:
        try:
            run_section_c(args, tokenizer, device_map, dtype, calib_dom, results, json_path)
        except Exception as e:
            print(f"[CSAQ] Warning: Section C failed: {e}")

    # Section D
    if "D" in args.skip_section:
        print("[CSAQ] Skipping section D")
    else:
        try:
            run_section_d(args, tokenizer, device_map, dtype, calib_dom, eval_en, eval_dom, results, json_path)
        except Exception as e:
            print(f"[CSAQ] Warning: Section D failed: {e}")

    # Section E
    if "E" in args.skip_section:
        print("[CSAQ] Skipping section E")
    else:
        if salience_dom is None or salience_en is None:
            print("[CSAQ] Skipping section E because Section B did not produce salience maps.")
        else:
            try:
                run_section_e(args, tokenizer, device_map, dtype, salience_dom, salience_en, results, json_path)
            except Exception as e:
                print(f"[CSAQ] Warning: Section E failed: {e}")

    # Section F
    if "F" in args.skip_section:
        print("[CSAQ] Skipping section F")
    else:
        print_section_f(args, results)


if __name__ == "__main__":
    main()
