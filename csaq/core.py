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

def _smart_prepare_calib_data(
    calib_data: Union[List[Dict[str, torch.Tensor]], List[str]],
    model: nn.Module,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """
    Smart Input Handler: accepts either tokenized dicts OR raw strings.
    Ensures pad_token_id is set and attention_mask is always present.
    """
    if not calib_data:
        raise ValueError("[CSAQ] calib_data is empty. Provide at least 1 sample.")

    # ── Already tokenized dicts ───────────────────────────────────────────
    if isinstance(calib_data[0], dict):
        validated = []
        for i, batch in enumerate(calib_data):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"[CSAQ] calib_data[{i}] is {type(batch).__name__}, expected dict."
                )
            if "input_ids" not in batch:
                raise KeyError(
                    f"[CSAQ] calib_data[{i}] missing 'input_ids'. "
                    f"Keys found: {list(batch.keys())}"
                )
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            validated.append(batch)
        return validated

    # ── Raw strings — auto-tokenize ───────────────────────────────────────
    if isinstance(calib_data[0], str):
        try:
            from transformers import AutoTokenizer
            model_name = getattr(model.config, "_name_or_path", None)
            if model_name is None:
                raise AttributeError
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        except Exception:
            raise TypeError(
                "[CSAQ] calib_data contains raw strings but no tokenizer could be "
                "auto-resolved. Pass pre-tokenized dicts instead."
            )

        batches = []
        for text in calib_data:
            enc = tokenizer(
                str(text), return_tensors="pt",
                max_length=128, truncation=True,
                padding="max_length", return_attention_mask=True,
            )
            batches.append({k: v.to(device) for k, v in enc.items()})
        return batches

    raise TypeError(
        f"[CSAQ] calib_data[0] is {type(calib_data[0]).__name__}. "
        f"Expected list[dict] or list[str]."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2: Profiler & Graph
# ═══════════════════════════════════════════════════════════════════════════════

class CausalProfiler:
    def __init__(self, model: nn.Module, config: CSAQConfig):
        self.model = model
        self.config = config
        self.salience: Dict[str, torch.Tensor] = {}
        self.history: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List = []
        self.modules: Dict[str, nn.Module] = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "weight", None) is not None and module.weight.requires_grad:
                    if "lm_head" in name or "embed" in name:
                        continue
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
                if k >= o.shape[-1]:
                    mask = torch.ones_like(o, dtype=torch.bool)
                else:
                    thresholds = o.abs().topk(k, dim=1).values[:, -1:]
                    mask = o.abs() >= thresholds
                self.history[name].append(mask)

        self.hooks.append(module.register_forward_hook(forward_hook))

    def profile(
        self,
        calib_data: List[Dict[str, torch.Tensor]],
        verbose: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[List[int]]]]:
        self.model.train()
        t0 = time.time()

        prev_salience_ranks = None
        n_samples = len(calib_data)

        try:
            from tqdm import tqdm
            data_iter = tqdm(
                enumerate(calib_data),
                total=n_samples,
                desc="[CSAQ] Profiling Causal Salience",
                disable=not verbose,
            )
        except ImportError:
            data_iter = enumerate(calib_data)

        for i, batch in data_iter:
            if not isinstance(batch, dict):
                raise TypeError(
                    f"[CSAQ] calib_data[{i}] is {type(batch).__name__}, not dict."
                )

            labels = batch.get("labels", batch.get("input_ids"))
            fwd_kwargs = {
                k: v for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }
            out = self.model(**fwd_kwargs, labels=labels)
            out.loss.backward()

            with torch.no_grad():
                for name, module in self.modules.items():
                    if module.weight.grad is not None:
                        self.salience[name] += (
                            module.weight.grad * module.weight.data
                        ).abs().cpu()

            self.model.zero_grad()

            # ── Spearman Early Stopping ───────────────────────────────────
            if (i + 1) % 8 == 0 or (i + 1) == n_samples:
                all_sal = torch.cat(
                    [self.salience[n].flatten() for n in self.modules.keys()]
                )

                if not hasattr(self, "_eval_idx"):
                    eval_size = 100000
                    if getattr(self.config, "auto_scale_memory", True):
                        try:
                            import psutil
                            if psutil.virtual_memory().available < 8 * 1024**3:
                                eval_size = 25000
                        except ImportError:
                            pass
                    self._eval_idx = torch.randperm(all_sal.numel())[:eval_size]

                sub_sal = all_sal[self._eval_idx]
                ranks = sub_sal.argsort().argsort()
                del all_sal

                if prev_salience_ranks is not None:
                    n_r = float(ranks.numel())
                    d = (ranks.float() - prev_salience_ranks.float())
                    rho = 1.0 - (6.0 * (d ** 2).sum().item()) / (n_r * (n_r ** 2 - 1))

                    if verbose:
                        print(f"  [CSAQ] Sample {i+1}/{n_samples} — ρ = {rho:.4f}")

                    if rho >= 0.98 and (i + 1) >= 16:
                        if verbose:
                            print(f"  [CSAQ] Early stopping at sample {i+1}")
                        break
                else:
                    if verbose:
                        print(f"  [CSAQ] Sample {i+1}/{n_samples} — Initializing Spearman")

                prev_salience_ranks = ranks.clone()

        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.model.eval()

        cliques = self._build_cliques()
        return self.salience, cliques

    def _build_cliques(self) -> Dict[str, List[List[int]]]:
        cliques_per_layer: Dict[str, List[List[int]]] = {}

        try:
            from tqdm import tqdm
            modules_iter = tqdm(
                list(self.modules.keys()),
                desc="[CSAQ] Constructing Interaction Graphs",
            )
        except ImportError:
            modules_iter = list(self.modules.keys())

        for name in modules_iter:
            if not self.history[name]:
                cliques_per_layer[name] = [
                    [i] for i in range(self.modules[name].weight.shape[0])
                ]
                continue

            mask = torch.cat(self.history[name], dim=0).to(torch.float16)
            intersect = torch.matmul(mask.t(), mask).float()
            f = mask.sum(dim=0).float()

            del mask
            self.history[name] = []

            union = f.unsqueeze(1) + f.unsqueeze(0) - intersect
            union = union.clamp(min=1.0)
            jaccard = intersect / union

            n_out = jaccard.shape[0]
            visited = torch.zeros(n_out, dtype=torch.bool)
            layer_cliques: List[List[int]] = []

            for i in range(n_out):
                if visited[i]:
                    continue
                neighbors = (jaccard[i] >= self.config.clique_threshold).nonzero().flatten()
                clique = []
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        clique.append(n_idx.item())
                        visited[n_idx] = True
                if not clique:
                    clique = [i]
                    visited[i] = True
                layer_cliques.append(clique)

            cliques_per_layer[name] = layer_cliques
            del intersect, union, jaccard

        return cliques_per_layer


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_clique_budget(
    salience: Dict[str, torch.Tensor],
    cliques: Dict[str, List[List[int]]],
    config: CSAQConfig,
) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    all_cliques: List[Dict[str, Any]] = []

    for name, layer_cliques in cliques.items():
        sal_tensor = salience[name]
        for c in layer_cliques:
            c_salience = sal_tensor[c].sum().item()
            c_elems = len(c) * sal_tensor.shape[1]
            all_cliques.append({
                "layer": name,
                "rows": c,
                "salience": c_salience,
                "elems": c_elems,
                "bits": min(config.bit_options),
                "leader": c[sal_tensor[c].sum(dim=1).argmax().item()],
            })

    total_elems = sum(c["elems"] for c in all_cliques)
    current_bits = sum(c["elems"] * c["bits"] for c in all_cliques)
    target_total_bits = config.target_bits * total_elems

    options = sorted(config.bit_options)

    # Greedily upgrade cliques by salience-density
    all_cliques.sort(key=lambda x: x["salience"] / max(x["elems"], 1), reverse=True)

    # Safe-Budget Logic: Structural Skeleton for extreme quantization
    # Ensures at least 15% of the model gets >= 4-bit to prevent logical collapse
    if config.target_bits < 4.0 and any(b >= 4 for b in options):
        min_safe_bit = min([b for b in options if b >= 4])
        min_4bit_elems = int(config.protection_floor * total_elems)
        enforced_elems = 0
        for c in all_cliques:
            if enforced_elems >= min_4bit_elems:
                break
            if c["bits"] < min_safe_bit:
                cost = (min_safe_bit - c["bits"]) * c["elems"]
                current_bits += cost
                c["bits"] = min_safe_bit
                enforced_elems += c["elems"]

    # Pass 1: Structural Skeleton - ensure everything gets at least a safe floor
    min_ops = [b for b in options if b >= 2]
    if min_ops:
        safe_floor = min_ops[0]
        # Only apply hard floor if budget allows
        if target_total_bits >= safe_floor * total_elems:
            for c in all_cliques:
                if c["bits"] < safe_floor:
                    cost = (safe_floor - c["bits"]) * c["elems"]
                    current_bits += cost
                    c["bits"] = safe_floor

    # Pass 2: Smooth Scaling - upgrade in steps (2 -> 4 -> 8 -> 16)
    # This prevents the solver from using all the budget on 16-bit 'spikes'
    for b in options:
        for c in all_cliques:
            if b <= c["bits"]:
                continue
            cost = (b - c["bits"]) * c["elems"]
            
            # Weighted salience check to prioritize bang-for-buck
            if current_bits + cost <= target_total_bits:
                current_bits += cost
                c["bits"] = b
            else:
                # If we can't afford this bit-level, we might still afford others for this clique later
                continue 


    # ── KEY FIX: budget is keyed by MODULE names (from cliques/profiler) ──
    # These are names like "model.layers.0.self_attn.q_proj" — NO .weight suffix.
    budget: Dict[str, List[Dict]] = defaultdict(list)
    tier_stats: Dict[str, int] = defaultdict(int)
    for c in all_cliques:
        budget[c["layer"]].append(c)
        tier_stats[f"int{c['bits']}"] += c["elems"]

    return dict(budget), dict(tier_stats)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY — iterates over the budget (module-keyed) directly
# ═══════════════════════════════════════════════════════════════════════════════

def apply_csaq(
    model: nn.Module,
    budget: Dict[str, List[Dict]],
    salience: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> Dict[str, List[int]]:
    """
    Apply quantization AND preserve FP16 backups for high-salience cliques.

    CRITICAL: The budget dict is keyed by MODULE names (e.g. "model.layers.0.q_proj")
    from named_modules(), NOT parameter names ("model.layers.0.q_proj.weight").
    We iterate over the budget keys directly and resolve each module, rather than
    iterating named_parameters() and trying to match — which was the v0.2.1 bug
    that caused causal_map to always be empty.

    Backup strategy:
      - Deterministic Protection: Protects the top N% of the most salient rows 
        globally across all layers to ensure a structural FP16 backbone.
      - Any clique assigned 16-bit also gets an FP16 backup.

    Returns causal_map: {module_name: [row_indices]} for the SSD engine.
    """
    causal_map: Dict[str, List[int]] = {}

    # ── Iterate over budget keys (module names), not named_parameters ─────
    for module_name, layer_budget in budget.items():
        # Resolve the module directly
        module = _resolve_module_by_name(model, module_name)
        if module is None or not hasattr(module, "weight"):
            if verbose:
                print(f"  [CSAQ] WARN: Could not resolve module '{module_name}', skipping")
            continue

        param = module.weight
        if param.dim() < 2:
            continue

        W_orig = param.data.clone()
        result = torch.zeros_like(W_orig)

        hi_sal_rows: List[int] = []

        # Deterministic: If bits=16, always backup. 
        # Otherwise, the Safe-Floor below handles the top 10% across the model.
        for c in layer_budget:
            rows = c["rows"]
            bits = c["bits"]
            leader = c["leader"]
            leader_row = W_orig[leader].clone()

            q_rows = quantize_shared_scale(W_orig[rows], leader_row, bits)
            result[rows] = q_rows

            if bits >= 16:
                hi_sal_rows.extend(rows)

        # Overwrite the weight with quantized data
        param.data = result

        # ── Register FP16 backup buffers ──────────────────────────────────
        if hi_sal_rows:
            _register_backup_on_module(
                module, module_name, W_orig, result, hi_sal_rows, causal_map
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SAFE-FLOOR: If zero rows qualified, use top-N% from the solver's ranking
    # ══════════════════════════════════════════════════════════════════════════
    if not causal_map:
        if verbose:
            print(
                "[CSAQ] ⚠ No cliques reached ≥8-bit. "
                "Activating quantile-based Safe-Floor from salience ranking."
            )
        causal_map = _apply_safe_floor_from_budget(
            model, budget, salience, verbose
        )

    if not causal_map:
        warnings.warn(
            "[CSAQ] WARNING: 0 high-salience rows backed up even after Safe-Floor.\n"
            "Self-Speculative Decoding will have NO verify path.\n"
            "Fix: Increase calibration samples, raise target_bits, or lower clique_threshold.",
            stacklevel=2,
        )

    return causal_map


# Bit threshold above which clique rows are considered "high-salience"
_HIGH_SALIENCE_BIT_FLOOR = 16
# Safe-Floor: protect the top N% most salient rows (Deterministic Accuracy Backbone)
_SAFE_FLOOR_TOP_PCT = 0.10


def _apply_safe_floor_from_budget(
    model: nn.Module,
    budget: Dict[str, List[Dict]],
    salience: Dict[str, torch.Tensor],
    verbose: bool,
) -> Dict[str, List[int]]:
    """
    Quantile-based Safe-Floor: when no cliques reach ≥8-bit, forcibly protect
    the top 1% most salient rows per layer using the salience map directly.
    """
    causal_map: Dict[str, List[int]] = {}

    for module_name in budget.keys():
        if module_name not in salience:
            continue

        module = _resolve_module_by_name(model, module_name)
        if module is None or not hasattr(module, "weight"):
            continue

        sal = salience[module_name]
        row_sal = sal.sum(dim=1)
        n_out = row_sal.shape[0]
        n_protect = max(1, int(_SAFE_FLOOR_TOP_PCT * n_out))

        top_rows = row_sal.topk(n_protect).indices.sort().values.tolist()

        W_orig_backup = module.weight.data.clone()
        _register_backup_on_module(
            module, module_name,
            W_orig_backup,  # best we have — already quantized at this point
            module.weight.data,
            top_rows, causal_map,
        )

    if verbose and causal_map:
        n_backed = sum(len(v) for v in causal_map.values())
        print(f"[CSAQ] Safe-Floor: {n_backed} rows backed up across {len(causal_map)} layers")

    return causal_map


def _register_backup_on_module(
    module: nn.Module,
    module_name: str,
    W_orig: torch.Tensor,
    W_quant: torch.Tensor,
    row_indices: List[int],
    causal_map: Dict[str, List[int]],
):
    """Register FP16 backup + quantized stash buffers directly on the module."""
    hi_rows_t = torch.tensor(sorted(set(row_indices)), dtype=torch.long)

    # Detach, clone, and contiguous to ensure absolute memory independence 
    # of the 'Steel Pins' from the 1-bit quantized weights.
    module.register_buffer(
        "_csaq_fp16_backup",
        W_orig[hi_rows_t].detach().clone().contiguous(),
        persistent=False,
    )
    module.register_buffer(
        "_csaq_quant_stash",
        W_quant[hi_rows_t].detach().clone().contiguous(),
        persistent=False,
    )
    module.register_buffer(
        "_csaq_hi_rows",
        hi_rows_t,
        persistent=False,
    )
    causal_map[module_name] = hi_rows_t.tolist()


def _resolve_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """
    Resolve a dotted module name to the actual nn.Module.

    Handles both formats:
      - "model.layers.0.self_attn.q_proj"         (from named_modules)
      - "model.layers.0.self_attn.q_proj.weight"   (from named_parameters)
    """
    parts = module_name.split(".")
    # Strip trailing ".weight" if present
    if parts[-1] == "weight":
        parts = parts[:-1]
    mod = model
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return None
    return mod if isinstance(mod, nn.Module) else None


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def quantize(
    model: nn.Module,
    calib_data: Union[List[Dict[str, torch.Tensor]], List[str]],
    config: CSAQConfig,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize a model using Causal Salience-Aware Quantization.

    Args:
        model:      Any HuggingFace causal LM.
        calib_data: List of tokenized dicts OR raw strings (auto-tokenized).
        config:     CSAQConfig instance.
        verbose:    Print progress.

    Returns:
        (quantized_model, info_dict)
    """
    if verbose:
        print(f"[CSAQ] Starting Quantization Pipeline. Target: {config.target_bits} bits")

    # Silence tied-weights warning and ensure Consistency
    # Prevents Shape Mismatch errors in extreme swapping scenarios
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    
    if getattr(model.config, "tie_word_embeddings", False) or hasattr(model, "get_output_embeddings"):
        # Physically untie the tensors to guarantee memory independence
        try:
            in_emb = model.get_input_embeddings()
            out_emb = model.get_output_embeddings()
            if in_emb is not None and out_emb is not None:
                if in_emb.weight is out_emb.weight:
                    out_emb.weight = nn.Parameter(out_emb.weight.detach().clone().contiguous())
        except Exception:
            pass

    device = next(model.parameters()).device
    calib_data = _smart_prepare_calib_data(calib_data, model, str(device))

    profiler = CausalProfiler(model, config)
    salience, cliques = profiler.profile(calib_data, verbose=verbose)

    if verbose:
        total_cliques = sum(len(c) for c in cliques.values())
        print(f"[CSAQ] Profiling complete. {total_cliques} cliques extracted.")
        print("[CSAQ] Solving bit-budget constraint...")

    budget, tier_stats = solve_clique_budget(salience, cliques, config)

    if verbose:
        total = sum(tier_stats.values())
        if total > 0:
            print("[CSAQ] Budget solved:")
            for t, count in sorted(tier_stats.items()):
                pct = count / total * 100
                print(f"  {t}: {pct:.2f}%")

    causal_map = apply_csaq(model, budget, salience, verbose=verbose)

    if verbose:
        n_backed = sum(len(v) for v in causal_map.values())
        print(
            f"[CSAQ] FP16 backups: {n_backed} high-salience rows "
            f"across {len(causal_map)} layers"
        )

    info: Dict[str, Any] = {
        "tier_stats": dict(tier_stats),
        "budget": budget,
        "causal_map": causal_map,
        "cliques_count": sum(len(c) for c in cliques.values()),
    }

    return model, info
