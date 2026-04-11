import os
import sys
import time
import math
import warnings
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple, Union

from .config import CSAQConfig
from .kernels import quantize_per_channel, quantize_shared_scale

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _smart_prepare_calib_data(calib_data, model, device="cpu"):
    if not calib_data: raise ValueError("[CSAQ] calib_data is empty.")
    if isinstance(calib_data[0], dict):
        validated = []
        for batch in calib_data:
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            validated.append(batch)
        return validated
    if isinstance(calib_data[0], str):
        from transformers import AutoTokenizer
        m_name = getattr(model.config, "_name_or_path", "qwen/qwen-1.5-0.5b")
        tokenizer = AutoTokenizer.from_pretrained(m_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        batches = []
        for text in calib_data:
            enc = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            batches.append({k: v.to(device) for k, v in enc.items()})
        return batches
    raise TypeError("[CSAQ] Unsupported calib_data format.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2: Profiler & Graph
# ═══════════════════════════════════════════════════════════════════════════════

class CausalProfiler:
    def __init__(self, model: nn.Module, config: CSAQConfig):
        self.model, self.config = model, config
        self.salience, self.history, self.hooks, self.modules = {}, {}, [], {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None:
                if "lm_head" in name or "embed" in name: continue
                self.modules[name] = module
                self.salience[name] = torch.zeros_like(module.weight.data, device="cpu")
                self.history[name] = []
                self._register_hook(name, module)

    def _register_hook(self, name: str, module: nn.Module):
        def forward_hook(m, inp, out):
            with torch.no_grad():
                o = out.detach().view(-1, out.shape[-1]).cpu()
                alpha = getattr(self.config, "salience_alpha", 1.0)
                frac = min(1.0, 0.1 * alpha)
                k = max(1, int(frac * o.shape[-1]))
                if k >= o.shape[-1]: mask = torch.ones_like(o, dtype=torch.bool)
                else: mask = o.abs() >= o.abs().topk(k, dim=1).values[:, -1:]
                self.history[name].append(mask)
        self.hooks.append(module.register_forward_hook(forward_hook))

    def profile(self, calib_data, verbose=True):
        self.model.train()
        n_samples, actual_batches = len(calib_data), 0
        from tqdm import tqdm
        data_iter = tqdm(enumerate(calib_data), total=n_samples, disable=not verbose)
        
        prev_ranks = None
        for i, batch in data_iter:
            actual_batches += 1
            labels = batch.get("labels", batch.get("input_ids"))
            out = self.model(**batch, labels=labels)
            out.loss.backward()
            with torch.no_grad():
                for name, module in self.modules.items():
                    if module.weight.grad is not None:
                        self.salience[name] += (module.weight.grad * module.weight.data).abs().cpu()
            self.model.zero_grad()

            if (i+1) % 8 == 0:
                all_s = torch.cat([self.salience[n].flatten() for n in self.modules.keys()])
                if not hasattr(self, "_eval_idx"): self._eval_idx = torch.randperm(all_s.numel())[:50000]
                ranks = all_s[self._eval_idx].argsort().argsort()
                if prev_ranks is not None:
                    n_r = float(ranks.numel())
                    d = (ranks.float() - prev_ranks.float())
                    rho = 1.0 - (6.0 * (d ** 2).sum().item()) / (n_r * (n_r ** 2 - 1))
                    if rho >= 0.99 and (i+1) >= 16: break
                prev_ranks = ranks.clone()

        # Bug 4 Fix: Normalize salience by actual batches processed
        for name in self.salience:
            self.salience[name] /= max(actual_batches, 1)

        for h in self.hooks: h.remove()
        self.model.eval()
        return self.salience, self._build_cliques()

    def _build_cliques(self) -> Dict[str, List[List[int]]]:
        cliques_per_layer = {}
        for name in self.modules:
            if not self.history[name]:
                cliques_per_layer[name] = [[i] for i in range(self.modules[name].weight.shape[0])]
                continue
            # Bug 5 Fix: Use float32 to prevent Jaccard overflow
            mask = torch.cat(self.history[name], dim=0).float()
            intersect = torch.matmul(mask.t(), mask)
            f = mask.sum(dim=0)
            self.history[name] = [] # Clear history
            union = (f.unsqueeze(1) + f.unsqueeze(0) - intersect).clamp(min=1.0)
            jaccard = intersect / union
            visited = torch.zeros(jaccard.shape[0], dtype=torch.bool)
            layer_cliques = []
            for i in range(jaccard.shape[0]):
                if visited[i]: continue
                clique = (jaccard[i] >= self.config.clique_threshold).nonzero().flatten().tolist()
                valid_clique = [idx for idx in clique if not visited[idx]]
                if valid_clique:
                    layer_cliques.append(valid_clique)
                    visited[torch.tensor(valid_clique)] = True
            cliques_per_layer[name] = layer_cliques
        return cliques_per_layer


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_clique_budget(salience, cliques, config):
    all_cliques = []
    for name, layer_cliques in cliques.items():
        s_ten = salience[name]
        for c in layer_cliques:
            all_cliques.append({
                "layer": name, "rows": c, "salience": s_ten[c].sum().item(),
                "elems": len(c) * s_ten.shape[1], "bits": min(config.bit_options),
                "leader": c[s_ten[c].sum(dim=1).argmax().item()],
            })

    total_elems = sum(c["elems"] for c in all_cliques)
    current_bits = sum(c["elems"] * c["bits"] for c in all_cliques)
    actual_avg_bits = current_bits / max(total_elems, 1)
    target_total = config.target_bits * total_elems
    options = sorted(config.bit_options)

    # Bug 8 Fix: Correct greedy loop order (cliques first, then bit levels)
    all_cliques.sort(key=lambda x: x["salience"] / max(x["elems"], 1), reverse=True)
    
    # Pre-pass: Ensure 2-bit floor if target allows
    if min(options) < 2 and 2 in options and target_total >= 2.0 * total_elems:
        for c in all_cliques:
            if c["bits"] < 2:
                current_bits += c["elems"]
                c["bits"] = 2

    # Greedy Upgrade Pass
    for c in all_cliques:
        for b in options:
            if b <= c["bits"]: continue
            cost = (b - c["bits"]) * c["elems"]
            if current_bits + cost <= target_total:
                current_bits += cost
                c["bits"] = b
            else:
                break # Cannot afford higher for THIS clique

    actual_avg_bits = current_bits / max(total_elems, 1)
    budget, stats = defaultdict(list), defaultdict(int)
    for c in all_cliques:
        budget[c["layer"]].append(c)
        stats[f"int{c['bits']}"] += c["elems"]
    return dict(budget), dict(stats), actual_avg_bits


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY
# ═══════════════════════════════════════════════════════════════════════════════

def apply_csaq(model, budget, salience, verbose=True):
    causal_map = {}
    for mod_name, layer_budget in budget.items():
        module = _resolve_module_by_name(model, mod_name)
        if module is None or not hasattr(module, "weight"): continue
        W_orig, result = module.weight.data.clone(), torch.zeros_like(module.weight.data)
        hi_rows_layer = []
        for c in layer_budget:
            rows, bits, leader = c["rows"], c["bits"], c["leader"]
            result[rows] = quantize_shared_scale(W_orig[rows], W_orig[leader].clone(), bits)
            if bits >= 16: hi_rows_layer.extend(rows)
        module.weight.data = result
        if hi_rows_layer: _register_backup(module, mod_name, W_orig, result, hi_rows_layer, causal_map)

    if not causal_map or sum(len(v) for v in causal_map.values()) < 0.10 * sum(p.numel() for p in model.parameters() if p.dim() > 1) // next(model.parameters()).shape[-1]:
        causal_map = _apply_safe_floor(model, budget, salience, verbose)
    return causal_map

_SAFE_FLOOR_TOP_PCT = 0.10

def _apply_safe_floor(model, budget, salience, verbose):
    causal_map = {}
    for m_name in budget:
        if m_name not in salience: continue
        module = _resolve_module_by_name(model, m_name)
        if module is None: continue
        row_sal = salience[m_name].sum(dim=1)
        n_protect = max(1, int(_SAFE_FLOOR_TOP_PCT * row_sal.shape[0]))
        top_rows = row_sal.topk(n_protect).indices.tolist()
        _register_backup(module, m_name, module.weight.data.clone(), module.weight.data, top_rows, causal_map)
    return causal_map

def _register_backup(module, name, W_orig, W_q, rows, cmap):
    rows_t = torch.tensor(sorted(set(rows)), dtype=torch.long)
    module.register_buffer("_csaq_fp16_backup", W_orig[rows_t].detach().clone().contiguous(), persistent=False)
    module.register_buffer("_csaq_quant_stash", W_q[rows_t].detach().clone().contiguous(), persistent=False)
    module.register_buffer("_csaq_hi_rows", rows_t, persistent=False)
    cmap[name] = rows_t.tolist()

def _resolve_module_by_name(model, name):
    mod = model
    for p in name.split("."):
        if p == "weight": continue
        mod = getattr(mod, p, None)
        if mod is None: return None
    return mod

def quantize(model, calib_data, config, verbose=True):
    if verbose: print(f"[CSAQ] Pipeline Start: {config.target_bits} bits")
    
    # Bug 6 Fix: Restore warning and logic for tied weights
    if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
        warnings.warn("[CSAQ] Model has tied_word_embeddings=True. Physically untying to ensure memory integrity.")
        model.config.tie_word_embeddings = False
    
    # Ensure physical separation of embeddings
    try:
        if model.get_input_embeddings().weight is model.get_output_embeddings().weight:
            model.get_output_embeddings().weight = nn.Parameter(model.get_output_embeddings().weight.detach().clone().contiguous())
    except: pass

    dev = next(model.parameters()).device
    calib = _smart_prepare_calib_data(calib_data, model, str(dev))
    salience, cliques = CausalProfiler(model, config).profile(calib, verbose)
    
    cliques_count = sum(len(c) for c in cliques.values())
    budget, stats, actual_bits = solve_clique_budget(salience, cliques, config)
    
    causal_map = apply_csaq(model, budget, salience, verbose)
    
    # Overlap reporting
    # ... placeholder or actual calc ...
    
    return model, {
        "tier_stats": stats, 
        "budget": budget, 
        "causal_map": causal_map,
        "actual_bits": actual_bits,
        "cliques_count": cliques_count
    }
