"""
csaq/core.py — Three-phase quantisation pipeline.

Phase 1: Causal salience profiling (gradient × activation)
Phase 2: Jaccard co-activation clique discovery
Phase 3: Fractional bit-budget constraint solving + weight application
"""

from __future__ import annotations

import math
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import CSAQConfig
from .kernels import (
    CSAQLinear,
    QuantizedWeight,
    inject_csaq_linear,
    quantize_per_channel,
    quantize_shared_scale,
    _get_submodule,
)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# layer_name → list of clique dicts
BudgetMap = Dict[str, List[Dict[str, Any]]]
# layer_name → salience tensor (out_features, in_features)
SalienceMap = Dict[str, torch.Tensor]
# layer_name → list of row-index lists
CliqueMap = Dict[str, List[List[int]]]


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_calib_data(
    calib_data: Union[List[Dict[str, torch.Tensor]], List[str]],
    model: nn.Module,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """
    Normalise calibration data to a list of tokenised batches.

    Accepts:
        - List of pre-tokenised dicts (``{"input_ids": ..., "attention_mask": ...}``)
        - List of raw strings (will be tokenised using the model's tokenizer)
    """
    if not calib_data:
        raise ValueError("[CSAQ] calib_data is empty.")

    if isinstance(calib_data[0], dict):
        validated: List[Dict[str, torch.Tensor]] = []
        for batch in calib_data:
            if "attention_mask" not in batch:
                batch = dict(batch)
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            validated.append({k: v.to(device) for k, v in batch.items()})
        return validated

    if isinstance(calib_data[0], str):
        from transformers import AutoTokenizer
        m_name = getattr(getattr(model, "config", None), "_name_or_path", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(m_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        batches: List[Dict[str, torch.Tensor]] = []
        for text in calib_data:
            enc = tokenizer(
                str(text),
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            batches.append({k: v.to(device) for k, v in enc.items()})
        return batches

    raise TypeError(
        "[CSAQ] Unsupported calib_data type.  Expected List[Dict] or List[str]."
    )


def _linear_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    """Yield (name, module) for all quantisable nn.Linear layers."""
    skip_patterns = ("lm_head", "embed_tokens", "embed_positions", "wte", "wpe")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight is None:
            continue
        if any(p in name for p in skip_patterns):
            continue
        yield name, module


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 & 2: CausalProfiler
# ─────────────────────────────────────────────────────────────────────────────

class CausalProfiler:
    """
    Profiles per-layer *causal salience* (∂L/∂w × w) and builds a
    Jaccard co-activation graph to find cliques of weight rows that
    fire together across calibration samples.

    Early stopping: once the Spearman rank correlation of accumulated
    salience between consecutive checkpoints exceeds 0.99, the ordering
    is considered stable and profiling halts — saving significant compute
    for large datasets.
    """

    _EARLY_STOP_RHO: float = 0.99
    _EARLY_STOP_MIN_BATCHES: int = 16
    _EVAL_STRIDE: int = 8          # check Spearman every N batches
    _SPEARMAN_SAMPLE: int = 50_000  # subsample for fast rank correlation

    def __init__(self, model: nn.Module, config: CSAQConfig) -> None:
        self.model = model
        self.config = config

        self.salience: SalienceMap = {}
        self._act_history: Dict[str, List[torch.Tensor]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._modules: Dict[str, nn.Linear] = {}

        for name, module in _linear_modules(model):
            self._modules[name] = module
            self.salience[name] = torch.zeros_like(module.weight.data, device="cpu")
            self._act_history[name] = []
            self._register_hook(name, module)

    def _register_hook(self, name: str, module: nn.Linear) -> None:
        alpha = self.config.salience_alpha

        def forward_hook(
            m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            with torch.no_grad():
                o = out.detach().view(-1, out.shape[-1]).cpu()
                frac = min(1.0, 0.1 * alpha)
                k = max(1, int(frac * o.shape[-1]))
                if k >= o.shape[-1]:
                    mask = torch.ones(o.shape[0], o.shape[-1], dtype=torch.bool)
                else:
                    threshold = o.abs().topk(k, dim=1).values[:, -1:]
                    mask = o.abs() >= threshold
                self._act_history[name].append(mask)

        self._hooks.append(module.register_forward_hook(forward_hook))

    def profile(
        self,
        calib_data: List[Dict[str, torch.Tensor]],
        verbose: bool = True,
    ) -> Tuple[SalienceMap, CliqueMap]:
        """
        Run calibration forward+backward passes to accumulate salience and
        activation co-occurrence data.

        Returns:
            Tuple of (salience_map, clique_map).
        """
        self.model.train()
        n_samples = len(calib_data)
        actual_batches = 0
        prev_ranks: Optional[torch.Tensor] = None
        _eval_idx: Optional[torch.Tensor] = None

        try:
            from tqdm import tqdm
            data_iter = tqdm(
                enumerate(calib_data),
                total=n_samples,
                desc="[CSAQ] Profiling",
                disable=not verbose,
            )
        except ImportError:
            data_iter = enumerate(calib_data)

        for i, batch in data_iter:
            actual_batches += 1
            labels = batch.get("labels", batch.get("input_ids"))
            try:
                out = self.model(**batch, labels=labels)
                out.loss.backward()
            except Exception as exc:
                warnings.warn(f"[CSAQ] Skipping calibration batch {i}: {exc}")
                self.model.zero_grad()
                continue

            with torch.no_grad():
                for name, module in self._modules.items():
                    if module.weight.grad is not None:
                        self.salience[name] += (
                            module.weight.grad * module.weight.data
                        ).abs().cpu()
            self.model.zero_grad()

            # Spearman early stopping
            if (i + 1) % self._EVAL_STRIDE == 0 and (i + 1) >= self._EARLY_STOP_MIN_BATCHES:
                all_s = torch.cat([self.salience[n].flatten() for n in self._modules])
                if _eval_idx is None:
                    size = min(self._SPEARMAN_SAMPLE, all_s.numel())
                    _eval_idx = torch.randperm(all_s.numel())[:size]
                ranks = all_s[_eval_idx].argsort().argsort().float()
                if prev_ranks is not None:
                    n_r = float(ranks.numel())
                    d = ranks - prev_ranks
                    rho = 1.0 - (6.0 * (d**2).sum().item()) / (n_r * (n_r**2 - 1))
                    if rho >= self._EARLY_STOP_RHO:
                        if verbose:
                            print(
                                f"[CSAQ] Early stop at batch {i+1}/{n_samples} "
                                f"(ρ={rho:.4f})"
                            )
                        break
                prev_ranks = ranks.clone()

        # Normalise by batches processed
        for name in self.salience:
            self.salience[name] = self.salience[name] / max(actual_batches, 1)

        for h in self._hooks:
            h.remove()
        self.model.eval()
        return self.salience, self._build_cliques()

    def _build_cliques(self) -> CliqueMap:
        """
        Greedily partition each layer's output channels into cliques using
        Jaccard similarity of their activation masks.

        Online bit-vector accumulation keeps memory O(channels²) per layer,
        not O(channels × samples).
        """
        cliques_per_layer: CliqueMap = {}

        for name, module in self._modules.items():
            history = self._act_history[name]
            n_channels = module.weight.shape[0]

            if not history:
                cliques_per_layer[name] = [[i] for i in range(n_channels)]
                continue

            # Accumulate intersection and union counts using float32
            mask = torch.cat(history, dim=0).float()   # (N_samples, channels)
            intersect = torch.matmul(mask.t(), mask)   # (C, C)
            f = mask.sum(dim=0)                        # (C,)
            union = (f.unsqueeze(1) + f.unsqueeze(0) - intersect).clamp(min=1.0)
            jaccard = intersect / union                # (C, C)
            self._act_history[name] = []               # free memory

            threshold = self.config.clique_threshold
            visited = torch.zeros(n_channels, dtype=torch.bool)
            layer_cliques: List[List[int]] = []

            for i in range(n_channels):
                if visited[i]:
                    continue
                similar = (jaccard[i] >= threshold).nonzero(as_tuple=False).flatten()
                clique = [idx.item() for idx in similar if not visited[idx]]
                if clique:
                    layer_cliques.append(clique)
                    visited[torch.tensor(clique, dtype=torch.long)] = True

            cliques_per_layer[name] = layer_cliques

        return cliques_per_layer


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Constraint solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_clique_budget(
    salience: SalienceMap,
    cliques: CliqueMap,
    config: CSAQConfig,
) -> Tuple[BudgetMap, Dict[str, int], float]:
    """
    Assign a bit-width to each clique such that the average bits-per-weight
    across the whole model is ≤ config.target_bits.

    Strategy
    ────────
    1. Start every clique at the minimum bit option.
    2. Apply the protection floor: force top-p% of salient cliques to ≥ 8-bit.
    3. Greedy upgrade: iterate cliques by descending salience density
       (salience / elements) and promote to the next bit tier whenever the
       additional bit cost fits within the remaining budget.

    Returns:
        (budget_map, tier_stats, actual_avg_bits)
    """
    all_cliques: List[Dict[str, Any]] = []
    options = sorted(config.bit_options)
    min_bits = options[0]

    for name, layer_cliques in cliques.items():
        s_ten = salience[name]
        for c in layer_cliques:
            row_sal = s_ten[c].sum(dim=1)       # (clique_size,)
            leader_idx = row_sal.argmax().item()
            all_cliques.append({
                "layer": name,
                "rows": c,
                "salience": s_ten[c].sum().item(),
                "elems": len(c) * s_ten.shape[1],
                "bits": min_bits,
                "leader": c[int(leader_idx)],
            })

    if not all_cliques:
        return {}, {}, float(config.target_bits)

    total_elems = sum(c["elems"] for c in all_cliques)
    current_bits = sum(c["elems"] * c["bits"] for c in all_cliques)
    target_total = config.target_bits * total_elems

    # Sort by salience density (descending) — highest value cliques upgraded first
    all_cliques.sort(key=lambda x: x["salience"] / max(x["elems"], 1), reverse=True)

    # Protection floor: top protection_floor fraction of cliques → min 8-bit
    n_protect = max(1, int(config.protection_floor * len(all_cliques)))
    for c in all_cliques[:n_protect]:
        floor_bits = max(c["bits"], min(8, max(options)))
        if floor_bits > c["bits"] and floor_bits in options:
            extra = (floor_bits - c["bits"]) * c["elems"]
            current_bits += extra
            c["bits"] = floor_bits

    # Greedy upgrade pass
    for c in all_cliques:
        for b in options:
            if b <= c["bits"]:
                continue
            cost = (b - c["bits"]) * c["elems"]
            if current_bits + cost <= target_total:
                current_bits += cost
                c["bits"] = b
            else:
                break

    actual_avg_bits = current_bits / max(total_elems, 1)

    # Assemble budget map
    budget: BudgetMap = defaultdict(list)
    tier_stats: Dict[str, int] = defaultdict(int)
    for c in all_cliques:
        budget[c["layer"]].append(c)
        tier_stats[f"int{c['bits']}"] += c["elems"]

    return dict(budget), dict(tier_stats), actual_avg_bits


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3b: Apply quantisation
# ─────────────────────────────────────────────────────────────────────────────

def apply_csaq(
    model: nn.Module,
    budget: BudgetMap,
    salience: SalienceMap,
    config: CSAQConfig,
    verbose: bool = True,
) -> Dict[str, List[int]]:
    """
    Apply quantisation to the model in two steps:

    1. Replace ``nn.Linear`` layers with ``CSAQLinear`` (packed storage).
       This is what actually saves memory — the original fp32 weight tensor
       is freed and replaced with a packed uint8 buffer.

    2. Register fp16 backups for high-salience rows so the speculative
       decoding engine can swap between draft (quantised) and verify (fp16)
       weights without a full dequantisation pass.

    Returns:
        causal_map: dict mapping layer names to lists of protected row indices.
    """
    # Step 1: inject CSAQLinear — real packing happens here
    model = inject_csaq_linear(model, budget, verbose=verbose)

    # Step 2: register speculative-decoding buffers on the new CSAQLinear layers
    causal_map: Dict[str, List[int]] = {}

    for mod_name, layer_cliques in budget.items():
        module = _get_submodule(model, mod_name)
        if not isinstance(module, CSAQLinear):
            continue
        if mod_name not in salience:
            continue

        row_sal = salience[mod_name].sum(dim=1)  # (out_features,)
        n_out = module.out_features
        n_protect = max(1, int(config.protection_floor * n_out))
        top_rows = row_sal.topk(n_protect).indices.sort().values

        # Dequantise just these rows into fp16 for the verify-pass backup
        W_fp32 = module._get_weight_fp32()
        fp16_backup = W_fp32[top_rows].to(torch.float16).detach().contiguous()
        quant_stash = W_fp32[top_rows].detach().contiguous()  # already quantised fp32

        module.register_buffer(
            "_csaq_hi_rows", top_rows, persistent=False
        )
        module.register_buffer(
            "_csaq_fp16_backup", fp16_backup, persistent=False
        )
        module.register_buffer(
            "_csaq_quant_stash", quant_stash, persistent=False
        )

        causal_map[mod_name] = top_rows.tolist()

    if verbose:
        n_protected = sum(len(v) for v in causal_map.values())
        print(f"[CSAQ] Protected {n_protected} rows across {len(causal_map)} layers for speculative decoding.")

    return causal_map


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def quantize(
    model: nn.Module,
    calib_data: Union[List[Dict[str, torch.Tensor]], List[str]],
    config: Optional[CSAQConfig] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Run the full CSAQ three-phase quantisation pipeline.

    Args:
        model:      A HuggingFace ``PreTrainedModel`` (or any ``nn.Module``
                    with ``nn.Linear`` layers and a ``model.forward(labels=...)``
                    that returns a loss).
        calib_data: Calibration samples — either pre-tokenised dicts
                    ``{"input_ids": Tensor, "attention_mask": Tensor}`` or
                    a list of raw strings.
        config:     :class:`~csaq.CSAQConfig`. Defaults to 4.0-bit target.
        verbose:    Print progress information.

    Returns:
        ``(quantized_model, info_dict)`` where ``info_dict`` contains:

        * ``tier_stats``    — dict mapping "int4" / "int8" etc. to element count
        * ``budget``        — the full clique budget map (for inspection)
        * ``causal_map``    — layer → protected row indices (for speculative decoding)
        * ``actual_bits``   — achieved average bits-per-weight
        * ``cliques_count`` — total number of weight cliques discovered
        * ``elapsed_s``     — total pipeline wall-clock time
    """
    if config is None:
        config = CSAQConfig()

    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CSAQ Quantization Pipeline")
        print(f"  Target: {config.target_bits} bits/weight")
        print(f"  Bit options: {config.bit_options}")
        print(f"  Clique threshold: {config.clique_threshold}")
        print(f"{'='*60}\n")

    # ── Handle tied embeddings ───────────────────────────────────────────────
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        warnings.warn(
            "[CSAQ] Model has tie_word_embeddings=True.  Untying to preserve "
            "memory integrity during quantisation.",
            stacklevel=2,
        )
        model.config.tie_word_embeddings = False
        try:
            if (
                model.get_input_embeddings().weight
                is model.get_output_embeddings().weight
            ):
                model.get_output_embeddings().weight = nn.Parameter(
                    model.get_output_embeddings().weight.detach().clone().contiguous()
                )
        except Exception:
            pass

    device = next(model.parameters()).device
    calib = _prepare_calib_data(calib_data, model, str(device))

    # ── Phase 1 & 2: Profiling ───────────────────────────────────────────────
    if verbose:
        print("[CSAQ] Phase 1+2: Causal salience profiling & clique discovery")
    profiler = CausalProfiler(model, config)
    salience, cliques = profiler.profile(calib, verbose=verbose)

    cliques_count = sum(len(v) for v in cliques.values())
    if verbose:
        print(f"[CSAQ] Discovered {cliques_count} weight cliques across {len(cliques)} layers.\n")

    # ── Phase 3a: Budget solver ──────────────────────────────────────────────
    if verbose:
        print("[CSAQ] Phase 3a: Bit-budget constraint solving")
    budget, tier_stats, actual_bits = solve_clique_budget(salience, cliques, config)
    if verbose:
        print(f"[CSAQ] Actual avg bits: {actual_bits:.3f} (target: {config.target_bits})")
        for tier, count in sorted(tier_stats.items()):
            pct = 100.0 * count / max(sum(tier_stats.values()), 1)
            print(f"  {tier}: {count:,} elements ({pct:.1f}%)")
        print()

    # ── Phase 3b: Apply & pack ───────────────────────────────────────────────
    if verbose:
        print("[CSAQ] Phase 3b: Applying quantisation & packing weights")
    causal_map = apply_csaq(model, budget, salience, config, verbose=verbose)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[CSAQ] Pipeline complete in {elapsed:.1f}s\n")

    return model, {
        "tier_stats": tier_stats,
        "budget": budget,
        "causal_map": causal_map,
        "actual_bits": actual_bits,
        "cliques_count": cliques_count,
        "elapsed_s": elapsed,
    }
