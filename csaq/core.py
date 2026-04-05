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
        self.history = {}
        self.hooks = []
        self.modules = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Only profile weights that require grad and avoid enormous unquantized heads
                if getattr(module, "weight", None) is not None and module.weight.requires_grad:
                    if "lm_head" in name or "embed" in name:
                        continue
                        
                    self.modules[name] = module
                    self.salience[name] = torch.zeros_like(module.weight.data, device="cpu")
                    self.history[name] = []
                    self._register_hook(name, module)
    
    def _register_hook(self, name, module):
        def forward_hook(m, inp, out):
            with torch.no_grad():
                o = out.detach().view(-1, out.shape[-1]).cpu() # (B*seq, n_out)
                # Top 10% Sparsification
                k = max(1, int(0.1 * o.shape[-1]))
                if k == o.shape[-1]:
                    mask = torch.ones_like(o, dtype=torch.bool)
                else:
                    thresholds = o.abs().topk(k, dim=1).values[:, -1:]
                    mask = o.abs() >= thresholds
                
                # Append boolean active mask (constant small memory footprint across L and V)
                self.history[name].append(mask)
                
        self.hooks.append(module.register_forward_hook(forward_hook))

    def profile(self, calib_data, verbose=True):
        self.model.train()
        t0 = time.time()
        
        prev_salience_ranks = None
        n_samples = len(calib_data)

        try:
            from tqdm import tqdm
            data_iter = tqdm(calib_data, desc="[CSAQ] Profiling Causal Salience", disable=not verbose)
        except ImportError:
            data_iter = calib_data

        for i, batch in enumerate(data_iter):
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
                # Concatenate all salience to compute proxy
                all_sal = torch.cat([self.salience[n].flatten() for n in self.modules.keys()])
                
                # Sample randomly to calculate the Spearman Rank 
                # Scales to RAM to prevent argsorts from consuming massive memory chunks dynamically.
                if not hasattr(self, "eval_idx"):
                     eval_size = 100000
                     if getattr(self.config, "auto_scale_memory", True):
                         import psutil
                         if psutil.virtual_memory().available < 8 * 1024**3:
                             eval_size = 25000
                     self.eval_idx = torch.randperm(all_sal.numel())[:eval_size]
                     
                sub_sal = all_sal[self.eval_idx]
                ranks = sub_sal.argsort().argsort() # rank of sub-sampled elements
                del all_sal
                
                if prev_salience_ranks is not None:
                    # Approximation of Spearman rho
                    n = float(ranks.numel())
                    d = (ranks.float() - prev_salience_ranks.float())
                    rho = 1.0 - (6.0 * (d ** 2).sum().item()) / (n * (n ** 2 - 1))
                    
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
        
        try:
            from tqdm import tqdm
            modules_iter = tqdm(list(self.modules.keys()), desc="[CSAQ] Constructing Graphs")
        except ImportError:
            modules_iter = self.modules.keys()
            
        for name in modules_iter:
            if not self.history[name]:
                cliques_per_layer[name] = [[i] for i in range(self.modules[name].weight.shape[0])]
                continue
                
            # Concatenate mask history (Total_N, n_out)
            mask = torch.cat(self.history[name], dim=0).to(torch.float16)  
            
            # Compute intersection efficiently for just this layer
            intersect = torch.matmul(mask.t(), mask).float()
            f = mask.sum(dim=0).float()
            
            # Free up the history to keep RAM clean
            del mask
            self.history[name] = []
            
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
            
            # Clean up the large NxN objects locally before moving to next layer
            del intersect, union, jaccard
            
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
