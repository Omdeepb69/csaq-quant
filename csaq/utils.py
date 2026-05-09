"""
csaq/utils.py — Calibration data, evaluation, reporting, and model export.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
    tokenizer: Any,
    n: int = 64,
    seq_len: int = 128,
    dataset: str = "wikitext",
    device: str = "cpu",
    hard: bool = False,
    custom_texts: Optional[List[str]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build a tokenised calibration dataset for CSAQ profiling.

    Args:
        tokenizer:    Any HuggingFace tokenizer with a ``__call__`` method.
        n:            Number of calibration samples.
        seq_len:      Sequence length to truncate/pad to.
        dataset:      ``"wikitext"`` (default) for general English;
                      ``"hard"`` for a mix of MATH + HumanEval + WikiText.
        device:       Device to place tensors on.
        hard:         If ``True``, override ``dataset`` with hard-domain data.
        custom_texts: If provided, tokenise these strings instead of loading
                      any public dataset.  Useful for domain-adapted quantisation.

    Returns:
        List of dicts ``{"input_ids": Tensor, "attention_mask": Tensor}``.
    """
    # User-supplied texts take priority
    if custom_texts is not None:
        return _tokenise_texts(tokenizer, custom_texts[:n], seq_len, device)

    texts: List[str] = []

    if hard or dataset == "hard":
        texts = _load_hard_texts(n)

    if not texts:
        texts = _load_wikitext(n)

    return _tokenise_texts(tokenizer, texts[:n], seq_len, device)


def _load_wikitext(n: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [t for t in ds["text"] if len(t.strip()) > 80][:n]


def _load_hard_texts(n: int) -> List[str]:
    texts: List[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", "main", split="test")
        texts += [
            f"Problem: {x['problem']}\nSolution: {x['solution']}"
            for x in list(ds)[: n // 3]
        ]
    except Exception:
        pass
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        texts += [x["prompt"] for x in list(ds)[: n // 3]]
    except Exception:
        pass
    if len(texts) < n:
        texts += _load_wikitext(n - len(texts))
    return texts


def _tokenise_texts(
    tokenizer: Any,
    texts: List[str],
    seq_len: int,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for text in texts:
        enc = tokenizer(
            str(text),
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    dataset: str = "wikitext2",
    max_tokens: int = 4096,
    stride: int = 512,
    seq_len: int = 128,
    device: str = "cpu",
) -> float:
    """
    Compute perplexity on WikiText-2 test split using a sliding-window approach.

    This is the standard PPL evaluation used by GPTQ, AWQ, and HQQ for fair
    comparison.  Use ``max_tokens=4096`` and ``stride=512`` to match their
    reported numbers.

    Args:
        model:      Quantised (or baseline) model.
        tokenizer:  Matching HuggingFace tokenizer.
        dataset:    Currently ``"wikitext2"`` (more datasets planned).
        max_tokens: Total tokens to evaluate over.
        stride:     Stride of the sliding window.
        seq_len:    Window size (context length per chunk).
        device:     Evaluation device.

    Returns:
        Perplexity score (lower is better).
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    inp = enc.input_ids[:, :max_tokens].to(device)

    nlls: List[float] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, inp.shape[1] - 1, stride):
            end = min(begin + seq_len, inp.shape[1])
            chunk = inp[:, begin:end]
            if chunk.shape[1] < 2:
                continue
            try:
                loss = model(chunk, labels=chunk).loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    nlls.append(loss.item())
            except Exception as exc:
                warnings.warn(f"[CSAQ] PPL eval chunk {begin}-{end} failed: {exc}")

    if not nlls:
        return float("inf")
    return float(np.exp(np.mean(nlls)))


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def generate_csaq_report(
    info: Dict[str, Any],
    save_path: str = "./CSAQ_Report.json",
) -> Dict[str, Any]:
    """
    Generate and save a JSON report from quantisation ``info`` dict.

    The report follows a stable schema so downstream tooling can parse it
    reliably.  Unknown keys in ``info`` are forwarded under ``"extra"``.

    Args:
        info:      The second return value from :func:`~csaq.core.quantize`.
        save_path: Output path for the JSON file.

    Returns:
        The report dict (also written to disk).
    """
    tier_stats = info.get("tier_stats", {})
    total_elems = sum(tier_stats.values()) if tier_stats else 1

    pct: Dict[str, float] = {
        t: round(100.0 * count / total_elems, 2)
        for t, count in tier_stats.items()
    }

    report: Dict[str, Any] = {
        "csaq_version": "0.5.0",
        "actual_avg_bits": round(info.get("actual_bits", 0.0), 4),
        "total_cliques": info.get("cliques_count", 0),
        "total_quantized_params": total_elems,
        "elapsed_seconds": round(info.get("elapsed_s", 0.0), 2),
        "bit_distribution": tier_stats,
        "bit_distribution_pct": pct,
        "salience_overlap_pct": info.get("overlap_pct", "not_computed"),
        "pareto_efficiency": info.get("pareto_score", "not_computed"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[CSAQ] Report saved → {save_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Model export
# ─────────────────────────────────────────────────────────────────────────────

def export_csaq_model(
    model: nn.Module,
    config: Any,        # CSAQConfig
    budget: Any,        # BudgetMap (optional, for manifest)
    save_path: str,
    info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a quantised CSAQ model to disk in a HuggingFace-compatible format.

    Layout::

        save_path/
            config.json          ← model + quantisation config (merged)
            csaq_manifest.json   ← clique metadata, version, bit stats
            model.safetensors    ← packed weight buffers

    The ``config.json`` includes a ``quantization_config`` block so that
    :class:`transformers.AutoModelForCausalLM` can detect the quantisation
    format.  A CSAQ-aware loader can then reconstruct ``CSAQLinear`` layers
    from the packed buffers.

    Args:
        model:     Quantised model (with ``CSAQLinear`` layers).
        config:    :class:`~csaq.CSAQConfig` used during quantisation.
        budget:    Clique budget map from ``info["budget"]``.
        save_path: Directory to write files into.
        info:      Full ``info`` dict from :func:`~csaq.core.quantize`
                   (used for manifest metadata).

    Returns:
        Absolute path to the saved directory.
    """
    import safetensors.torch

    os.makedirs(save_path, exist_ok=True)

    # ── 1. config.json ─────────────────────────────────────────────────────
    # Merge model config + quantisation config for HF compatibility
    try:
        model_config_dict = model.config.to_dict()
    except AttributeError:
        model_config_dict = {}

    quant_config = config.to_dict()
    quant_config["quant_type"] = "csaq"
    quant_config["csaq_version"] = "0.5.0"

    # HF convention: quantisation_config is a nested key
    model_config_dict["quantization_config"] = quant_config

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config_dict, f, indent=2)

    # ── 2. csaq_manifest.json ──────────────────────────────────────────────
    causal_map_serialisable: Dict[str, Any] = {}
    if info:
        raw_cmap = info.get("causal_map", {})
        for k, v in raw_cmap.items():
            causal_map_serialisable[k] = v if isinstance(v, list) else v.tolist()

    manifest: Dict[str, Any] = {
        "csaq_version": "0.5.0",
        "bit_distribution": info.get("tier_stats", {}) if info else {},
        "actual_avg_bits": info.get("actual_bits", 0.0) if info else 0.0,
        "cliques_count": info.get("cliques_count", 0) if info else 0,
        "causal_map": causal_map_serialisable,
        "target_bits": config.target_bits,
        "bit_options": config.bit_options,
        "group_size": config.group_size,
    }
    with open(
        os.path.join(save_path, "csaq_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    # ── 3. model.safetensors ───────────────────────────────────────────────
    # Non-persistent buffers (fp16 backups for speculative decoding) are
    # intentionally excluded from state_dict and won't be saved here.
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.endswith(("_csaq_fp16_backup", "_csaq_quant_stash", "_csaq_hi_rows"))
    }
    safetensors.torch.save_file(
        state_dict,
        os.path.join(save_path, "model.safetensors"),
    )

    abs_path = os.path.abspath(save_path)
    print(f"[CSAQ] Model exported → {abs_path}")
    print(f"         config.json, csaq_manifest.json, model.safetensors")
    return abs_path
