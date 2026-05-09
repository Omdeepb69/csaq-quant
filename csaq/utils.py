"""
csaq/utils.py — Calibration data, evaluation, reporting, and model export.

Design principle: NO default dataset is assumed.  The caller always provides
``custom_texts``.  This ensures quantisation reflects the actual target domain —
Konkani, medical text, code, or anything else — not English Wikipedia.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_data(
    tokenizer: Any,
    custom_texts: List[str],
    n: Optional[int] = None,
    seq_len: int = 128,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """
    Build a tokenised calibration dataset from **your own texts**.

    There is intentionally no default dataset.  The salience map is only
    meaningful relative to the domain you care about.  If you quantise a
    Konkani model using English Wikipedia as calibration, CSAQ will protect
    English-important weights, not Konkani ones.

    Args:
        tokenizer:    Any HuggingFace tokenizer.
        custom_texts: List of strings in your target language/domain.
                      Recommended minimum: 64 samples of >= 50 tokens each.
        n:            If set, use only the first ``n`` texts.
                      Defaults to all texts supplied.
        seq_len:      Token length to truncate/pad each sample to.
        device:       PyTorch device string (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        List of dicts ``{"input_ids": Tensor, "attention_mask": Tensor}``.

    Konkani example::

        texts = [
            "आमी कोंकणी उलयतात.",
            "कोंकणी भाशा भारताची राष्ट्रीय भाशा.",
            # ... at least 64 sentences recommended
        ]
        calib = build_calibration_data(tokenizer, custom_texts=texts)
        model, info = quantize(model, calib, config=config)
    """
    if not custom_texts:
        raise ValueError(
            "[CSAQ] custom_texts is empty.\n"
            "Provide sentences in your target language/domain.\n"
            "Example:\n"
            "  calib = build_calibration_data(tokenizer,\n"
            "              custom_texts=['Your sentence 1', 'Your sentence 2', ...])"
        )

    texts = list(custom_texts) if n is None else list(custom_texts)[:n]

    if len(texts) < 16:
        warnings.warn(
            f"[CSAQ] Only {len(texts)} calibration samples supplied. "
            "Salience estimates will be noisy. Recommended minimum: 64.",
            stacklevel=2,
        )

    return _tokenise_texts(tokenizer, texts, seq_len, device)


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
    eval_texts: List[str],
    max_tokens: int = 4096,
    stride: int = 512,
    seq_len: int = 128,
    device: str = "cpu",
) -> float:
    """
    Compute perplexity on **your own evaluation texts**.

    No dataset is loaded automatically.  Pass texts in the language/domain
    you are evaluating.  For comparison with GPTQ/AWQ published numbers,
    pass WikiText-2 test texts explicitly.  For Konkani evaluation, pass
    Konkani test sentences.

    The texts are joined with ``"\\n\\n"`` and evaluated with a sliding window
    — the standard protocol used in GPTQ, AWQ, and HQQ papers.

    Args:
        model:       Model to evaluate (quantised or baseline).
        tokenizer:   Matching HuggingFace tokenizer.
        eval_texts:  List of evaluation strings in your target domain.
        max_tokens:  Total tokens to evaluate across the joined text.
        stride:      Sliding-window stride (512 matches GPTQ/AWQ protocol).
        seq_len:     Window/chunk size.
        device:      Evaluation device.

    Returns:
        Perplexity (lower is better).
    """
    if not eval_texts:
        raise ValueError("[CSAQ] eval_texts is empty.")

    text = "\n\n".join(eval_texts)
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
                warnings.warn(f"[CSAQ] PPL chunk {begin}-{end} failed: {exc}")

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
    Generate and save a JSON report from the quantisation ``info`` dict.

    Args:
        info:      Second return value from :func:`~csaq.core.quantize`.
        save_path: Output path for the JSON file.

    Returns:
        The report dict (also written to disk).
    """
    tier_stats = info.get("tier_stats", {})
    total_elems = sum(tier_stats.values()) if tier_stats else 0

    pct: Dict[str, float] = (
        {t: round(100.0 * c / total_elems, 2) for t, c in tier_stats.items()}
        if total_elems > 0 else {}
    )

    report: Dict[str, Any] = {
        "csaq_version": "0.5.1",
        "actual_avg_bits": round(info.get("actual_bits", 0.0), 4),
        "total_cliques": info.get("cliques_count", 0),
        "total_quantized_params": total_elems,
        "elapsed_seconds": round(info.get("elapsed_s", 0.0), 2),
        "bit_distribution": tier_stats,
        "bit_distribution_pct": pct,
        "ppl": info.get("ppl", "not_computed"),
        "calibration_domain": info.get("calibration_domain", "user_provided"),
        "salience_overlap_pct": info.get("overlap_pct", "not_computed"),
        "memory_before_gb": info.get("memory_before_gb", "not_measured"),
        "memory_after_gb": info.get("memory_after_gb", "not_measured"),
        "memory_saved_gb": info.get("memory_saved_gb", "not_measured"),
        "memory_saved_pct": info.get("memory_saved_pct", "not_measured"),
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
    config: Any,
    budget: Any,
    save_path: str,
    info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a quantised CSAQ model to disk in a HuggingFace-compatible format.

    Output layout::

        save_path/
            config.json          ← model + quantisation config (merged)
            csaq_manifest.json   ← clique metadata, version, bit stats
            model.safetensors    ← packed weight buffers

    The ``config.json`` contains a ``quantization_config`` block so that
    HuggingFace ``AutoModelForCausalLM`` can detect the quantisation format.

    Returns:
        Absolute path to the saved directory.
    """
    import safetensors.torch

    os.makedirs(save_path, exist_ok=True)

    # 1. config.json
    try:
        model_config_dict = model.config.to_dict()
    except AttributeError:
        model_config_dict = {}

    config.base_model_type = model_config_dict.get("model_type", "unknown")
    config.base_model_name_or_path = model_config_dict.get("_name_or_path", model_config_dict.get("name_or_path", "unknown"))

    quant_config = config.to_dict()
    quant_config["quant_type"] = "csaq"
    quant_config["csaq_version"] = "0.5.1"
    
    model_config_dict["model_type"] = "csaq"
    model_config_dict.update(quant_config)
    if "quantization_config" in model_config_dict:
        del model_config_dict["quantization_config"]

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config_dict, f, indent=2)
        
    tokenizer_config_path = os.path.join(save_path, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    # 2. csaq_manifest.json
    causal_map_s: Dict[str, Any] = {}
    if info:
        for k, v in info.get("causal_map", {}).items():
            causal_map_s[k] = v if isinstance(v, list) else v.tolist()

    layer_bits: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, __import__("csaq.kernels", fromlist=["CSAQLinear"]).CSAQLinear):
            layer_bits[name] = module.bits

    manifest: Dict[str, Any] = {
        "csaq_version": "0.5.1",
        "bit_distribution": info.get("tier_stats", {}) if info else {},
        "actual_avg_bits": info.get("actual_bits", 0.0) if info else 0.0,
        "cliques_count": info.get("cliques_count", 0) if info else 0,
        "causal_map": causal_map_s,
        "target_bits": config.target_bits,
        "bit_options": config.bit_options,
        "group_size": config.group_size,
        "calibration_domain": info.get("calibration_domain", "user_provided") if info else "user_provided",
        "layer_bits": layer_bits,
    }
    with open(os.path.join(save_path, "csaq_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 3. model.safetensors — exclude non-persistent speculative buffers
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if not any(k.endswith(s) for s in (
            "_csaq_fp16_backup", "_csaq_quant_stash", "_csaq_hi_rows"
        ))
    }
    safetensors.torch.save_file(
        state_dict,
        os.path.join(save_path, "model.safetensors"),
    )

    abs_path = os.path.abspath(save_path)
    print(f"[CSAQ] Model exported → {abs_path}")
    return abs_path
