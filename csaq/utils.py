"""
csaq/utils.py — calibration data builders, evaluation helpers, and reporting
"""

import torch
import numpy as np
import json
import os
from typing import Optional, List, Dict

def build_calibration_data(
    tokenizer,
    n:          int = 64,
    seq_len:    int = 128,
    dataset:    str = "wikitext",
    device:     str = "cpu",
    hard:       bool = False,
) -> List[Dict[str, torch.Tensor]]:
    from datasets import load_dataset
    texts = []
    if hard:
        try:
            ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
            texts += [f"Problem: {x['problem']}\nSolution: {x['solution']}" for x in list(ds)[:n//3]]
        except Exception:
            pass
        try:
            ds = load_dataset("openai_humaneval", split="test")
            texts += [x["prompt"] for x in list(ds)[:n//3]]
        except Exception:
            pass
        if len(texts) < n:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            texts += [t for t in ds["text"][5000:] if len(t.strip()) > 80][:n-len(texts)]
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 80]

    batches = []
    for text in texts:
        enc = tokenizer(
            str(text), return_tensors="pt",
            max_length=seq_len, truncation=True, padding="max_length"
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
        if len(batches) >= n:
            break
    return batches

def compute_perplexity(
    model,
    tokenizer,
    max_tokens: int = 4096,
    stride:     int = 512,
    seq_len:    int = 128,
    device:     str = "cpu",
) -> float:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    inp = enc.input_ids[:, :max_tokens].to(device)

    nlls = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, inp.shape[1] - 1, stride):
            end = min(begin + seq_len, inp.shape[1])
            chunk = inp[:, begin:end]
            if chunk.shape[1] < 2:
                continue
            loss = model(chunk, labels=chunk).loss
            nlls.append(loss.item())

    return float(np.exp(np.mean(nlls)))

def generate_csaq_report(info: Dict, save_path: str = "./CSAQ_Report.json"):
    """Generate metric logging for CSAQ profiling output."""
    tier_stats = info.get("tier_stats", {})
    total_elems = sum(tier_stats.values()) if tier_stats else 1
    
    # Calculate overlap using actual dataset or placeholder fallback
    overlap = info.get("overlap_pct", "Not Computed")
    
    report = {
        "Salience_Magnitude_Overlap_Pct": overlap,
        "Bit_Distribution_Histogram": {
            t: count for t, count in tier_stats.items()
        },
        "Pareto_Efficiency_Score": info.get("pareto_score", "Not Computed"),
        "Total_Cliques": info.get("cliques_count", 0),
        "Quantized_Params": total_elems
    }
    
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"[CSAQ] Report saved to {save_path}")

def export_csaq_model(model, config, budget, save_path: str):
    """Export model to safetensors keeping Hugging Face compatibility in mind."""
    import safetensors.torch
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save config with architecture updates
    config_dict = config.to_dict()
    # Save the budget details inside the config file or logic mappings
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
        
    # 2. Serialize weights
    state_dict = model.state_dict()
    
    # 3. Add explicit buffer for the leader logic if necessary (simplified)
    # The actual implementation of Dequantization loads normal safetensors, 
    # but the custom class CSAQForCausalLM would interpret it correctly.
    safetensors.torch.save_file(state_dict, os.path.join(save_path, "model.safetensors"))
    
    # Additionally save the constraint mappings side-by-side
    with open(os.path.join(save_path, "csaq_clique_map.json"), "w") as f:
        json.dump(budget, f, indent=4)
        
    print(f"[CSAQ] Model exported via safetensors to {save_path}")
