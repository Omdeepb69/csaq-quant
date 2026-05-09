import subprocess
import sys
import traceback

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "csaq-quant>=0.5.9", "scipy", "matplotlib", "psutil"])
print("All dependencies installed.")

import os
import json
import time
import math
import psutil
import torch
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from csaq import CSAQConfig, quantize
from csaq.core import CausalProfiler
from csaq.inference import CSAQInferenceEngine
from csaq.utils import build_calibration_data, compute_perplexity

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen1.5-0.5B"
CORPUS_PATHS = [
    "/kaggle/input/datasets/omdeep22/hindi-small/hindi-small.txt",
    "./hindi_corpus.txt",
    "hindi_corpus.txt"
]

RESULTS = {}
SALIENCE_MAP_STORE = {}
SAVE_PATH = "/kaggle/working/csaq_research_results.json"
FIG_DIR = "/kaggle/working/figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("/kaggle/working", exist_ok=True)

def save_results():
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

def measure_mem() -> float:
    """VRAM tracking (includes CUDA context overhead)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024**3)
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def model_weight_memory_gb(model) -> float:
    """Measure actual bytes used by model parameters and registered buffers
       (not CUDA context overhead). Excludes speculative decoding backups."""
    total_bytes = 0
    for name, param in model.named_parameters():
        total_bytes += param.nelement() * param.element_size()
    for name, buf in model.named_buffers():
        if not any(x in name for x in [
            '_csaq_fp16_backup', '_csaq_quant_stash', '_csaq_hi_rows'
        ]):
            total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024**3)

print("Loading corpus...")
corpus_lines = []
for p in CORPUS_PATHS:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            corpus_lines = [line.strip() for line in f if line.strip()]
        break

if len(corpus_lines) < 30:
    print(f"ERROR: Found {len(corpus_lines)} lines in corpus, expected at least 30.")
    sys.exit(1)

CALIB_TEXTS = corpus_lines[:70]
EVAL_TEXTS = corpus_lines[70:]
EVAL_TEXTS_PPL = EVAL_TEXTS[:50]

print(f"Corpus loaded: {len(CALIB_TEXTS)} calibration lines, {len(EVAL_TEXTS)} evaluation lines")
if CALIB_TEXTS:
    print(f"First calib line: {CALIB_TEXTS[0][:60]}")

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def fresh_model():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    return model

# ==============================================================================
# SECTION A — PPL BENCHMARK
# ==============================================================================
try:
    print("\n" + "="*70 + "\nSECTION A — PPL BENCHMARK\n" + "="*70)
    RESULTS["section_A"] = {}
    
    # A1. FP32 baseline
    print("Loading FP32 baseline...")
    model = fresh_model()
    mem_base = measure_mem()
    ppl_base = compute_perplexity(model, tokenizer, eval_texts=EVAL_TEXTS_PPL, max_tokens=2048, stride=256, seq_len=128, device=DEVICE)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    RESULTS["section_A"]["ppl_base"] = ppl_base
    RESULTS["section_A"]["mem_base"] = mem_base
    
    # A2. CSAQ 4-bit
    print("Quantising to 4-bit...")
    model = fresh_model()
    mem_before_4b = measure_mem()
    calib_4b = build_calibration_data(tokenizer, custom_texts=CALIB_TEXTS, seq_len=128, device=DEVICE)
    config_4b = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16], clique_threshold=0.85, protection_floor=0.10)
    model, info_4b = quantize(model, calib_4b, config=config_4b, verbose=True, calibration_domain="hindi")
    mem_after_4b = measure_mem()
    mem_saved_4b = mem_before_4b - mem_after_4b
    ppl_4bit = compute_perplexity(model, tokenizer, eval_texts=EVAL_TEXTS_PPL, max_tokens=2048, stride=256, seq_len=128, device=DEVICE)
    actual_bits_4b = info_4b.get("actual_bits", 4.0)
    cliques_count_4b = info_4b.get("cliques_count", 0)
    tier_stats_4b = info_4b.get("tier_stats", {})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    RESULTS["section_A"]["ppl_4bit"] = ppl_4bit
    RESULTS["section_A"]["mem_saved_4b"] = mem_saved_4b
    RESULTS["section_A"]["actual_bits_4b"] = actual_bits_4b
    RESULTS["section_A"]["cliques_count"] = cliques_count_4b
    RESULTS["section_A"]["tier_stats"] = tier_stats_4b
    
    # A3. CSAQ 8-bit
    print("Quantising to 8-bit...")
    model = fresh_model()
    mem_before_8b = measure_mem()
    calib_8b = build_calibration_data(tokenizer, custom_texts=CALIB_TEXTS, seq_len=128, device=DEVICE)
    config_8b = CSAQConfig(target_bits=8.0, bit_options=[8, 16], clique_threshold=0.85, protection_floor=0.10)
    model, info_8b = quantize(model, calib_8b, config=config_8b, verbose=True, calibration_domain="hindi")
    mem_after_8b = measure_mem()
    mem_saved_8b = mem_before_8b - mem_after_8b
    ppl_8bit = compute_perplexity(model, tokenizer, eval_texts=EVAL_TEXTS_PPL, max_tokens=2048, stride=256, seq_len=128, device=DEVICE)
    actual_bits_8b = info_8b.get("actual_bits", 8.0)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    RESULTS["section_A"]["ppl_8bit"] = ppl_8bit
    RESULTS["section_A"]["mem_saved_8b"] = mem_saved_8b
    RESULTS["section_A"]["actual_bits_8b"] = actual_bits_8b
    
    def format_pct(base, val):
        return f"+{((val / base) - 1) * 100:.1f}%" if base > 0 else "N/A"
    
    print("\n+---------------------------+--------+--------+---------+----------+")
    print("| Config                    | Bits   | PPL    | vs FP32 | Mem Saved|")
    print("+---------------------------+--------+--------+---------+----------+")
    print(f"| FP32 baseline             |  32.00 | {ppl_base:>6.3f} |    —    |    —     |")
    print(f"| CSAQ 4-bit [Hindi calib]  | {actual_bits_4b:>6.2f} | {ppl_4bit:>6.3f} | {format_pct(ppl_base, ppl_4bit):>7} | {mem_saved_4b:>5.2f}gb |")
    print(f"| CSAQ 8-bit [Hindi calib]  | {actual_bits_8b:>6.2f} | {ppl_8bit:>6.3f} | {format_pct(ppl_base, ppl_8bit):>7} | {mem_saved_8b:>5.2f}gb |")
    print("+---------------------------+--------+--------+---------+----------+")
    
    print(f"\nActual cliques discovered: {cliques_count_4b}")
    print(f"Bit distribution: {tier_stats_4b}")
    
    save_results()
except Exception as e:
    print(f"Section A failed:")
    traceback.print_exc()

# ==============================================================================
# SECTION B — JACCARD CLIQUE ANALYSIS
# ==============================================================================
try:
    print("\n" + "="*70 + "\nSECTION B — JACCARD CLIQUE ANALYSIS\n" + "="*70)
    RESULTS["section_B"] = {}
    
    model = fresh_model()
    calib_data = build_calibration_data(tokenizer, custom_texts=CALIB_TEXTS, seq_len=128, device=DEVICE)
    config_j = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16], clique_threshold=0.85)
    
    profiler = CausalProfiler(model, config_j)
    salience_map, clique_map = profiler.profile(calib_data, verbose=True)
    SALIENCE_MAP_STORE.update(salience_map)
    
    all_sizes = []
    layer_stats = []
    total_cliques = 0
    all_rhos = []
    
    for layer_name, layer_cliques in clique_map.items():
        total_cliques += len(layer_cliques)
        sizes = [len(c) for c in layer_cliques]
        all_sizes.extend(sizes)
        singleton_count = sum(1 for s in sizes if s == 1)
        mean_size = float(np.mean(sizes)) if sizes else 0.0
        max_size = max(sizes) if sizes else 0
        
        rhos = []
        for c in layer_cliques:
            if len(c) >= 2:
                rows = salience_map[layer_name][c]
                pairs = []
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        pairs.append((i, j))
                
                if len(pairs) > 6:
                    idx = np.random.choice(len(pairs), 6, replace=False)
                    sampled = [pairs[p] for p in idx]
                else:
                    sampled = pairs
                    
                for pi, pj in sampled:
                    rho, _ = scipy.stats.spearmanr(rows[pi].cpu().numpy(), rows[pj].cpu().numpy())
                    if not np.isnan(rho):
                        rhos.append(rho)
                        all_rhos.append(rho)
                        
        layer_rho = float(np.mean(rhos)) if rhos else 0.0
        layer_stats.append({
            "name": layer_name,
            "cliques": len(layer_cliques),
            "singletons": singleton_count,
            "mean_size": mean_size,
            "max_size": max_size,
            "rho": layer_rho
        })
    
    mean_intra_rho = float(np.mean(all_rhos)) if all_rhos else 0.0
    mean_clique_size = float(np.mean(all_sizes)) if all_sizes else 0.0
    
    RESULTS["section_B"]["total_cliques"] = total_cliques
    RESULTS["section_B"]["mean_clique_size"] = mean_clique_size
    RESULTS["section_B"]["mean_intra_rho"] = mean_intra_rho
    
    print("\n+--------------------------------+---------+------------+-----------+-------+")
    print("| Layer                          | Cliques | Singletons | Mean size | rho   |")
    print("+--------------------------------+---------+------------+-----------+-------+")
    for stat in layer_stats[:8]:
        name_trunc = stat['name'][-30:] if len(stat['name']) > 30 else stat['name']
        print(f"| {name_trunc:<30} | {stat['cliques']:>7} | {stat['singletons']:>10} | {stat['mean_size']:>9.2f} | {stat['rho']:>5.2f} |")
    print("+--------------------------------+---------+------------+-----------+-------+")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # B2. Ablation
    print("\nRunning Ablation...")
    model = fresh_model()
    model, _ = quantize(model, calib_data, config=config_j, verbose=False, calibration_domain="hindi")
    ppl_jaccard = compute_perplexity(model, tokenizer, eval_texts=EVAL_TEXTS_PPL, max_tokens=2048, stride=256, seq_len=128, device=DEVICE)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    model = fresh_model()
    # Force per-channel: threshold > 1.0 makes all singletons,
    # bit_options=[4] prevents the solver from escaping to 16-bit
    config_p = CSAQConfig(target_bits=4.0, bit_options=[4], clique_threshold=1.01)
    model, info_pc = quantize(model, calib_data, config=config_p, verbose=False, calibration_domain="hindi")
    print(f"  Per-channel ablation: actual_bits={info_pc.get('actual_bits', '?')}, "
          f"cliques={info_pc.get('cliques_count', '?')}")
    ppl_perchannel = compute_perplexity(model, tokenizer, eval_texts=EVAL_TEXTS_PPL, max_tokens=2048, stride=256, seq_len=128, device=DEVICE)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    jaccard_gain = ppl_perchannel - ppl_jaccard
    RESULTS["section_B"]["ppl_jaccard"] = ppl_jaccard
    RESULTS["section_B"]["ppl_perchannel"] = ppl_perchannel
    RESULTS["section_B"]["jaccard_gain"] = jaccard_gain
    
    print(f"Jaccard clique PPL:       {ppl_jaccard:.3f}")
    print(f"Per-channel baseline PPL: {ppl_perchannel:.3f}")
    print(f"Clique grouping benefit:  {jaccard_gain:.3f} PPL points")
    if ppl_jaccard < ppl_perchannel:
        print("HYPOTHESIS SUPPORTED")
    else:
        print("Inconclusive at this scale")
        
    # B3. Reproducibility
    model = fresh_model()
    calib_shuffled = []
    # shuffle order
    idx = torch.randperm(len(calib_data)).tolist()
    calib_shuffled = [calib_data[i] for i in idx]
    profiler2 = CausalProfiler(model, config_j)
    salience_map2, _ = profiler2.profile(calib_shuffled, verbose=False)
    
    rho_values_rep = []
    for layer in list(salience_map.keys())[:20]:  # sample 20 layers
        if layer not in salience_map2:
            continue
        f1 = salience_map[layer].flatten().cpu().numpy()
        f2 = salience_map2[layer].flatten().cpu().numpy()
        k = min(5000, len(f1))
        rho_l, _ = scipy.stats.spearmanr(f1[:k], f2[:k])
        if not np.isnan(rho_l):
            rho_values_rep.append(rho_l)
    rho_rep = float(np.mean(rho_values_rep)) if rho_values_rep else 0.0

    print(f"Salience reproducibility rho: {rho_rep:.4f}")
    RESULTS["section_B"]["salience_reproducibility"] = float(rho_rep)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    save_results()
    
    # Figure 2: clique_analysis.png
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    ax1.hist(all_sizes, bins=range(1, max(all_sizes)+2), align='left', alpha=0.7)
    ax1.axvline(mean_clique_size, color='r', linestyle='dashed', linewidth=1.5)
    ax1.set_title("Clique Size Distribution")
    ax1.set_xlabel("Size")
    ax1.set_ylabel("Count")
    
    tier_stats = RESULTS.get("section_A", {}).get("tier_stats", {})
    if tier_stats:
        tiers = [str(k) for k in tier_stats.keys()]
        counts = [v / 1e6 for v in tier_stats.values()]
        ax2.bar(tiers, counts, alpha=0.7, color='g')
    else:
        ax2.text(0.5, 0.5, "Section A data not available",
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Bit Distribution Across Layers")
    ax2.set_xlabel("Bit Tier")
    ax2.set_ylabel("Elements (Millions)")
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "clique_analysis.png"))
    plt.show()
    
except Exception as e:
    print(f"Section B failed:")
    traceback.print_exc()

# ==============================================================================
# SECTION C — MEMORY BENCHMARK
# ==============================================================================
try:
    print("\n" + "="*70 + "\nSECTION C — MEMORY BENCHMARK\n" + "="*70)
    RESULTS["section_C"] = {}
    
    mem_rows = []
    calib_data = build_calibration_data(tokenizer, custom_texts=CALIB_TEXTS, seq_len=128, device=DEVICE)
    
    BIT_CONFIGS = {
        32.0: None,  # no quantisation
        8.0:  CSAQConfig(target_bits=8.0, bit_options=[8, 16],
                         clique_threshold=0.85),
        4.0:  CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16],
                         clique_threshold=0.85),
    }
    
    fp32_measured = 0.0
    for bits in [32.0, 8.0, 4.0]:
        model = fresh_model()
        total_params = sum(p.numel() for p in model.parameters())
        
        if bits < 32.0:
            config = BIT_CONFIGS[bits]
            model, _ = quantize(model, calib_data, config=config, verbose=False, calibration_domain="hindi")
            
        mem_after = model_weight_memory_gb(model)
        theoretical_gb = total_params * bits / 8 / 1e9
        fp32_size_gb = total_params * 32 / 8 / 1e9
        
        if bits == 32.0:
            fp32_measured = mem_after
            saved = "—"
            saved_gb = 0.0
            compression = 1.0
            mem_rows.append(["FP32", f"{total_params/1e6:.2f}M", theoretical_gb, mem_after, saved, f"{compression:.2f}x"])
            RESULTS["section_C"]["fp32_size"] = mem_after
        else:
            saved_gb = max(0.0, fp32_measured - mem_after)
            compression = fp32_measured / max(mem_after, 1e-9)
            saved = f"{saved_gb:.2f}gb"
            mem_rows.append([f"INT{int(bits)}", f"{total_params/1e6:.2f}M", theoretical_gb, mem_after, saved, f"{compression:.2f}x"])
            RESULTS["section_C"][f"int{int(bits)}_size"] = mem_after
            RESULTS["section_C"][f"int{int(bits)}_saved"] = saved_gb
            
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print("Note: Measured GB = model parameter + buffer bytes only")
    print("      (excludes CUDA context ~1.2GB fixed overhead)")
    print("+----------+----------+----------------+-------------+--------+---------+")
    print("| Config   | Params   | Theoretical GB | Measured GB | Saved  | Ratio   |")
    print("+----------+----------+----------------+-------------+--------+---------+")
    for r in mem_rows:
        print(f"| {r[0]:<8} | {r[1]:<8} | {r[2]:>14.3f} | {r[3]:>11.3f} | {r[4]:>6} | {r[5]:>7} |")
    print("+----------+----------+----------------+-------------+--------+---------+")
    
    save_results()
except Exception as e:
    print(f"Section C failed:")
    traceback.print_exc()

# ==============================================================================
# SECTION D — SPECULATIVE DECODING BENCHMARK
# ==============================================================================
try:
    print("\n" + "="*70 + "\nSECTION D — SPECULATIVE DECODING BENCHMARK\n" + "="*70)
    RESULTS["section_D"] = {}
    
    model = fresh_model()
    calib_data = build_calibration_data(tokenizer, custom_texts=CALIB_TEXTS, seq_len=128, device=DEVICE)
    config = CSAQConfig(target_bits=4.0, bit_options=[4, 8, 16], clique_threshold=0.85)
    model, info = quantize(model, calib_data, config=config, verbose=False, calibration_domain="hindi")
    
    # Verify model produces finite logits before benchmarking
    test_ids = tokenizer.encode("test", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        test_logits = model(test_ids).logits
    if not torch.isfinite(test_logits).all():
        print("WARNING: Model produces non-finite logits.")
        print("Skipping Section D — fix quantisation first.")
        raise RuntimeError("Non-finite logits detected")
    del test_ids, test_logits
    
    engine = CSAQInferenceEngine(model, info["causal_map"], tokenizer=tokenizer, verbose=False)
    engine.warmup(n=2)
    
    test_prompts = [p for p in EVAL_TEXTS if len(p.split()) >= 3][:3]
    test_prompts = [" ".join(p.split()[:10]) for p in test_prompts]
    
    print("+--------------------------------+--------+---------+---------+---------+")
    print("| Prompt (first 30 chars)        | Mode   | Accept  | tok/s   | Speedup |")
    print("+--------------------------------+--------+---------+---------+---------+")
    
    speedups_by_la = {4: [], 6: [], 8: []}
    acc_by_la = {4: [], 6: [], 8: []}
    
    std_tps_all = []
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        p_trunc = prompt[:30].ljust(30)
        
        # Standard
        std_tps_runs = []
        for _ in range(4):
            _, rep = engine.generate(input_ids, speculative=False, max_new_tokens=32, temperature=0.0)
            std_tps_runs.append(rep.tokens_per_second)
        std_tps = np.mean(std_tps_runs[1:])
        std_tps_all.append(float(std_tps))
        
        print(f"| {p_trunc} | Std    |    —    | {std_tps:>7.1f} |  1.00x  |")
        
        # Speculative
        for la in [4, 6, 8]:
            tps_runs = []
            acc_runs = []
            for _ in range(4):
                _, rep = engine.generate(input_ids, speculative=True, lookahead=la, max_new_tokens=32, temperature=0.0)
                tps_runs.append(rep.tokens_per_second)
                acc_runs.append(rep.acceptance_rate)
            tps = np.mean(tps_runs[1:])
            acc = np.mean(acc_runs[1:])
            speedup = tps / std_tps
            
            speedups_by_la[la].append(speedup)
            acc_by_la[la].append(acc)
            
            print(f"| {p_trunc} | la={la:<2} | {acc*100:>6.1f}% | {tps:>7.1f} | {speedup:>5.2f}x |")
            
    print("+--------------------------------+--------+---------+---------+---------+")
    
    RESULTS["section_D"]["std_tps_mean"] = float(np.mean(std_tps_all))
    
    for la in [4, 6, 8]:
        RESULTS["section_D"][f"speedup_la{la}"] = float(np.mean(speedups_by_la[la]))
        RESULTS["section_D"][f"acc_la{la}"] = float(np.mean(acc_by_la[la]))
        print(f"Mean speedup at la={la}: {RESULTS['section_D'][f'speedup_la{la}']:.2f}x")
        
    all_acc = []
    for la in [4, 6, 8]: all_acc.extend(acc_by_la[la])
    RESULTS["section_D"]["mean_acc"] = float(np.mean(all_acc))
    print(f"Mean acceptance rate: {RESULTS['section_D']['mean_acc']*100:.1f}%")
    
    del model, engine
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    save_results()
    
    # Figure 4: speculative_speedup.png
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    lookaheads = [4, 6, 8]
    speeds = [RESULTS["section_D"][f"speedup_la{la}"] for la in lookaheads]
    ax.plot(lookaheads, speeds, marker='o', color='b', linewidth=2)
    ax.axhline(1.0, color='r', linestyle='dashed')
    
    for i, la in enumerate(lookaheads):
        acc_pct = RESULTS["section_D"][f"acc_la{la}"] * 100
        ax.annotate(f"{acc_pct:.1f}%", (la, speeds[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        
    ax.set_title("Self-Speculative Decoding Speedup vs Lookahead")
    ax.set_xlabel("Lookahead")
    ax.set_ylabel("Speedup over Standard Generation")
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "speculative_speedup.png"))
    plt.show()
    
except Exception as e:
    print(f"Section D failed:")
    traceback.print_exc()

# ==============================================================================
# SECTION E — SALIENCE TOPOLOGY
# ==============================================================================
try:
    print("\n" + "="*70 + "\nSECTION E — SALIENCE TOPOLOGY\n" + "="*70)
    
    if SALIENCE_MAP_STORE:
        sal_map = SALIENCE_MAP_STORE
        
        largest_layer = max(
            sal_map.keys(),
            key=lambda k: sal_map[k].numel()
        )
        
        s_full = sal_map[largest_layer].cpu().numpy()
        n_out, n_in = s_full.shape
        
        r_idx = np.arange(0, n_out, max(1, n_out // 64))[:64]
        c_idx = np.arange(0, n_in, max(1, n_in // 64))[:64]
        
        s_sampled = s_full[r_idx][:, c_idx]
        s_sampled = np.log1p(s_sampled) # normalize
        
        row_sums = s_sampled.sum(axis=1)
        sorted_indices = np.argsort(row_sums)[::-1]
        s_sorted = s_sampled[sorted_indices]
        
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        
        im1 = ax1.imshow(s_sampled, cmap='hot', aspect='auto')
        ax1.set_title("Unsorted Salience")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        im2 = ax2.imshow(s_sorted, cmap='hot', aspect='auto')
        ax2.set_title("Rows Sorted by Descending Salience")
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        fig.suptitle(f"Salience Topology: {largest_layer}")
        fig.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "salience_topology.png"))
        plt.show()
        
        RESULTS["section_E"] = {"completed": True, "layer": largest_layer}
        save_results()
    else:
        print("Salience map not found, skipping section E.")
except Exception as e:
    print(f"Section E failed:")
    traceback.print_exc()

# ==============================================================================
# SECTION F — PAPER SUMMARY BLOCK
# ==============================================================================
try:
    # Figure 1: ppl_comparison.png (Build here so we use all stats)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ppl_b = RESULTS.get("section_A", {}).get("ppl_base", 0)
    ppl_4 = RESULTS.get("section_A", {}).get("ppl_4bit", 0)
    ppl_8 = RESULTS.get("section_A", {}).get("ppl_8bit", 0)
    
    labels = ["FP32", "CSAQ 8-bit", "CSAQ 4-bit"]
    vals = [ppl_b, ppl_8, ppl_4]
    
    ax.bar(labels, vals, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.8)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
    
    if ppl_b > 0:
        ax.axhline(ppl_b, color='r', linestyle='dashed', alpha=0.5)
        
    ax.set_title("Perplexity by Quantisation Config (Hindi eval)")
    ax.set_ylabel("Perplexity")
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ppl_comparison.png"))
    plt.show()

    def get_res(s, k, default="N/A"):
        return RESULTS.get(s, {}).get(k, default)
        
    def safe_pct_change(base, new):
        try:
            if base is None or new is None: return "N/A"
            if not isinstance(base, (int, float)): return "N/A"
            if not isinstance(new, (int, float)): return "N/A"
            if base == 0: return "N/A"
            return f"+{((new/base)-1)*100:.1f}%"
        except:
            return "N/A"

    def safe_fmt(v, fmt_str="{:.3f}"):
        try:
            if v is None or v == "N/A": return "N/A"
            return fmt_str.format(float(v))
        except:
            return "N/A"

    p_base = get_res("section_A", "ppl_base")
    p_4b = get_res("section_A", "ppl_4bit")
    p_8b = get_res("section_A", "ppl_8bit")
    deg_4b_pct = safe_pct_change(p_base, p_4b)
    deg_8b_pct = safe_pct_change(p_base, p_8b)
    
    try:
        deg_4b_pts = safe_fmt(p_4b - p_base, "{:.3f}") if p_base != "N/A" and p_4b != "N/A" else "N/A"
    except:
        deg_4b_pts = "N/A"

    mem_fp32 = get_res("section_C", "fp32_size")
    mem_4b = get_res("section_C", "int4_size")
    mem_saved = get_res("section_C", "int4_saved")
    
    try:
        mem_saved_pct = safe_fmt((mem_saved / mem_fp32)*100, "{:.1f}") if mem_fp32 != "N/A" and mem_fp32 > 0 and mem_saved != "N/A" else "N/A"
    except:
        mem_saved_pct = "N/A"
        
    try:
        comp_ratio = safe_fmt(mem_fp32 / mem_4b, "{:.2f}") if mem_fp32 != "N/A" and mem_4b != "N/A" and mem_4b > 0 else "N/A"
    except:
        comp_ratio = "N/A"

    best_la = 8
    best_spd = 0
    for la in [4,6,8]:
        s = get_res("section_D", f"speedup_la{la}")
        if s != "N/A" and s > best_spd:
            best_spd = s
            best_la = la
            
    best_tps = get_res("section_D", "std_tps_mean")
    
    if best_tps != "N/A" and best_spd != "N/A" and best_tps:
        best_spec_tps = best_tps * best_spd
    else:
        best_spec_tps = "N/A"

    print("\n  ================================================================")
    print("   CSAQ Research Paper — Key Results")
    print("   Model : Qwen/Qwen1.5-0.5B")
    print(f"   Domain: Hindi  |  Calibration: {len(CALIB_TEXTS)} sentences")
    print("  ================================================================")
    print("\n  [Table 1 — Quantisation Quality (Hindi eval)]")
    print(f"  FP32 baseline PPL            : {safe_fmt(p_base)}")
    print(f"  CSAQ 4-bit PPL               : {safe_fmt(p_4b)}  ({deg_4b_pct} degradation)")
    print(f"  CSAQ 8-bit PPL               : {safe_fmt(p_8b)}  ({deg_8b_pct} degradation)")
    print(f"  4-bit PPL degradation        : {deg_4b_pts} points")

    print("\n  [Table 2 — Clique Structure]")
    print(f"  Total cliques discovered     : {get_res('section_B', 'total_cliques')}")
    print(f"  Mean clique size             : {safe_fmt(get_res('section_B', 'mean_clique_size'), '{:.2f}')} channels")
    print(f"  Mean intra-clique salience rho: {safe_fmt(get_res('section_B', 'mean_intra_rho'), '{:.4f}')}")
    print(f"  Salience reproducibility rho : {safe_fmt(get_res('section_B', 'salience_reproducibility'), '{:.4f}')}")
    print(f"  Jaccard vs per-channel gain  : {safe_fmt(get_res('section_B', 'jaccard_gain'))} PPL points")

    print("\n  [Table 3 — Memory Savings]")
    print(f"  FP32 model size              : {safe_fmt(mem_fp32, '{:.2f}')} GB")
    print(f"  CSAQ 4-bit size              : {safe_fmt(mem_4b, '{:.2f}')} GB")
    print(f"  Memory saved (4-bit)         : {safe_fmt(mem_saved, '{:.2f}')} GB  ({mem_saved_pct}%)")
    print(f"  Compression ratio            : {comp_ratio}x")

    print("\n  [Table 4 — Speculative Decoding]")
    print(f"  Standard tok/s               : {safe_fmt(best_tps, '{:.1f}')}")
    print(f"  Best speculative tok/s       : {safe_fmt(best_spec_tps, '{:.1f}')}  (la={best_la})")
    print(f"  Best speedup                 : {safe_fmt(best_spd, '{:.2f}')}x")
    acc = get_res("section_D", "mean_acc")
    print(f"  Mean acceptance rate         : {safe_fmt(acc*100 if acc != 'N/A' else 'N/A', '{:.1f}')}%")

    print("\n  ================================================================")
    print("  All results saved to /kaggle/working/csaq_research_results.json")
    print("  All figures saved to /kaggle/working/figures/")
    print("  ================================================================\n")
    
    save_results()

except Exception as e:
    print(f"Section F failed:")
    traceback.print_exc()
