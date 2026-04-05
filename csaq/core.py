import torch
import torch.nn as nn
from collections import defaultdict
import time
import math
from typing import Optional, List, Dict, Any, Tuple
from .config import CSAQConfig
from .kernels import quantize_per_channel, quantize_shared_scale

# --- PHASE 1 & 2: Profiler & Graph ---

class CausalProfiler:
    def __init__(self, model, config: CSAQConfig):
        self.model = model
        self.config = config
        self.salience = {}
        self.intersection = {}
        self.freqs = {}
        self.hooks = []
        self.modules = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # We only profile weights that require grad
                if getattr(module, "weight", None) is not None and module.weight.requires_grad:
                    self.modules[name] = module
                    self.salience[name] = torch.zeros_like(module.weight.data, device="cpu")
                    
                    n_out = module.weight.shape[0]
                    self.intersection[name] = torch.zeros((n_out, n_out), dtype=torch.long, device="cpu")
                    self.freqs[name] = torch.zeros((n_out,), dtype=torch.long, device="cpu")
                    self._register_hook(name, module)
    
    def _register_hook(self, name, module):
        def forward_hook(m, inp, out):
            # out: (batch, seq, out_features)
            with torch.no_grad():
                o = out.detach().view(-1, out.shape[-1]).cpu() # (B*seq, n_out)
                # Top 10% Sparsification
                k = max(1, int(0.1 * o.shape[-1]))
                if k == o.shape[-1]:
                    mask = torch.ones_like(o, dtype=torch.bool)
                else:
                    thresholds = o.abs().topk(k, dim=1).values[:, -1:]
                    mask = o.abs() >= thresholds
                
                # Active mask is (N, n_out)
                mask = mask.to(torch.float16)  # float16 for fast matmul
                intersect = torch.matmul(mask.t(), mask).to(torch.long)
                
                self.intersection[name] += intersect
                self.freqs[name] += mask.sum(dim=0).to(torch.long)
                
        self.hooks.append(module.register_forward_hook(forward_hook))

    def profile(self, calib_data, verbose=True):
        self.model.train()
        t0 = time.time()
        
        prev_salience_ranks = None
        n_samples = len(calib_data)

        for i, batch in enumerate(calib_data):
            labels = batch.get("labels", batch.get("input_ids"))
            out = self.model(**{k: v for k, v in batch.items()
                                if k in ("input_ids", "attention_mask", "labels")},
                             labels=labels)
            out.loss.backward()

            with torch.no_grad():
                for name, module in self.modules.items():
                    if module.weight.grad is not None:
                        self.salience[name] += (module.weight.grad * module.weight.data).abs().cpu()
            
            self.model.zero_grad()

            # Spearman Early Stopping every 8 samples
            if (i + 1) % 8 == 0 or (i + 1) == n_samples:
                # Concatenate all salience to compute global rank
                all_sal = torch.cat([self.salience[n].flatten() for n in self.modules.keys()])
                # Sort to get ranks
                ranks = all_sal.argsort().argsort() # rank of each element
                
                if prev_salience_ranks is not None:
                    # Approximation of Spearman rho avoiding large float64 sums
                    # Since n is huge (~Billions), calculating full Pearson on ranks can be slow
                    # We compute exact formula: 1 - 6 sum(d^2) / (n(n^2 - 1))
                    n = float(ranks.numel())
                    # To avoid overflow, sample if too large
                    if n > 100000:
                        idx = torch.randperm(int(n))[:100000]
                        d = (ranks[idx].float() - prev_salience_ranks[idx].float())
                        n_sub = 100000.0
                    else:
                        d = (ranks.float() - prev_salience_ranks.float())
                        n_sub = n
                        
                    rho = 1.0 - (6.0 * (d ** 2).sum().item()) / (n_sub * (n_sub ** 2 - 1))
                    
                    if verbose:
                        print(f"  [CSAQ] Sample {i+1}/{n_samples} - Spearman rho: {rho:.4f}")
                        
                    if rho >= 0.98 and (i + 1) >= 16:
                        if verbose:
                            print(f"  [CSAQ] Early stopping triggered at sample {i+1}!")
                        break
                else:
                    if verbose:
                        print(f"  [CSAQ] Sample {i+1}/{n_samples} - Initializing Spearman")
                prev_salience_ranks = ranks.clone()

        for h in self.hooks:
            h.remove()
        self.model.eval()
        
        # Build cliques
        cliques = self._build_cliques()
        return self.salience, cliques
    
    def _build_cliques(self):
        cliques_per_layer = {}
        for name in self.modules.keys():
            intersect = self.intersection[name].float()
            f = self.freqs[name].float()
            union = f.unsqueeze(1) + f.unsqueeze(0) - intersect
            union = union.clamp(min=1.0)
            jaccard = intersect / union
            
            n_out = jaccard.shape[0]
            visited = torch.zeros(n_out, dtype=torch.bool)
            
            layer_cliques = []
            
            # Find cliques greedily
            for i in range(n_out):
                if visited[i]: continue
                
                # Jaccard above threshold
                neighbors = (jaccard[i] >= self.config.clique_threshold).nonzero().flatten()
                
                clique = []
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        clique.append(n_idx.item())
                        visited[n_idx] = True
                        
                if len(clique) == 0:
                    clique = [i]
                    visited[i] = True
                    
                layer_cliques.append(clique)
            cliques_per_layer[name] = layer_cliques
        return cliques_per_layer

# --- PHASE 3: Solver ---

def solve_clique_budget(salience: Dict[str, torch.Tensor], cliques: Dict[str, List[List[int]]], config: CSAQConfig):
    # Flatten cliques into sortable structs
    all_cliques = []
    
    for name, layer_cliques in cliques.items():
        sal_tensor = salience[name]
        for c in layer_cliques:
            # Salience of the clique is the sum of salience of its parameters
            # Each row is an output neuron.
            c_salience = sal_tensor[c].sum().item()
            c_elems = len(c) * sal_tensor.shape[1]
            all_cliques.append({
                "layer": name,
                "rows": c,
                "salience": c_salience,
                "elems": c_elems,
                "bits": min(config.bit_options),
                "leader": c[sal_tensor[c].sum(dim=1).argmax().item()]
            })

    total_elems = sum(c["elems"] for c in all_cliques)
    current_bits = sum(c["elems"] * c["bits"] for c in all_cliques)
    target_total_bits = target_avg_bits = config.target_bits * total_elems
    
    options = sorted(config.bit_options)
    
    # Greedily upgrade cliques by efficiency
    all_cliques.sort(key=lambda x: x["salience"] / x["elems"], reverse=True)
    
    for c in all_cliques:
        for b in options:
            if b <= c["bits"]: continue
            cost = (b - c["bits"]) * c["elems"]
            if current_bits + cost <= target_total_bits:
                current_bits += cost
                c["bits"] = b
            else:
                break
                
    # Group results by layer
    budget = defaultdict(list)
    tier_stats = defaultdict(int)
    for c in all_cliques:
        budget[c["layer"]].append(c)
        tier_stats[f"int{c['bits']}"] += c["elems"]
        
    return budget, tier_stats

# --- APPLY ---

def apply_csaq(model, budget: Dict[str, List[Dict]], verbose=True):
    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2 or name not in budget:
            continue
            
        W = param.data.clone()
        result = torch.zeros_like(W)
        
        layer_budget = budget[name]
        for c in layer_budget:
            rows = c["rows"]
            bits = c["bits"]
            leader = c["leader"]
            # To apply shared scale, leader row must be passed
            leader_row = W[leader].clone()
            
            q_rows = quantize_shared_scale(W[rows], leader_row, bits)
            result[rows] = q_rows
            
        param.data = result

# --- ENTRY ---

def quantize(model, calib_data, config: CSAQConfig, verbose=True):
    if verbose:
        print(f"[CSAQ] Starting Quantization Pipeline. Target: {config.target_bits} bits")
    
    profiler = CausalProfiler(model, config)
    salience, cliques = profiler.profile(calib_data, verbose=verbose)
    
    if verbose:
        print("[CSAQ] Profiling completed. Solving budget...")
        
    budget, tier_stats = solve_clique_budget(salience, cliques, config)
    
    if verbose:
        total = sum(tier_stats.values())
        print(f"[CSAQ] Applied:")
        for t, count in tier_stats.items():
            print(f"  {t}: {count/total*100:.1f}%")
            
    apply_csaq(model, budget, verbose=verbose)
    
    info = {
        "tier_stats": dict(tier_stats),
        "budget": budget,
        "cliques_count": sum(len(c) for c in cliques.values())
    }
    
    return model, info
