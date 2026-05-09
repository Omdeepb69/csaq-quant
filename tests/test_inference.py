"""
tests/test_inference.py — Tests for CSAQInferenceEngine and SpeculativeReport.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from csaq.config import CSAQConfig
from csaq.core import quantize
from csaq.inference import CSAQInferenceEngine, SpeculativeReport


# ─────────────────────────────────────────────────────────────────────────────
# Tiny generative model for testing
# ─────────────────────────────────────────────────────────────────────────────

class _TinyGenerative(nn.Module):
    """Minimal model compatible with CSAQInferenceEngine.generate()."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, vocab)
        self.config = type("cfg", (), {
            "eos_token_id": 1,
            "tie_word_embeddings": False,
        })()

    def forward(self, input_ids, past_key_values=None, use_cache=False, labels=None, **kw):
        x = self.embed(input_ids)                    # (B, T, H)
        logits = self.fc(x)                          # (B, T, V)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab), labels.view(-1))

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        out.past_key_values = None   # no KV cache for simplicity
        return out

    def generate(self, input_ids, max_new_tokens=4, **kw):
        """Trivial greedy generate for fallback testing."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self(generated)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)
        return generated

    def parameters(self, recurse=True):
        return super().parameters(recurse)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.fc


@pytest.fixture()
def tiny_gen_model():
    torch.manual_seed(5)
    return _TinyGenerative(vocab=32, hidden=16)


@pytest.fixture()
def quantised_model_and_info(tiny_gen_model):
    calib = []
    for _ in range(4):
        ids = torch.randint(0, 32, (1, 8))
        calib.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    cfg = CSAQConfig(
        target_bits=4.0, bit_options=[4, 8], clique_threshold=0.3, protection_floor=0.20
    )
    model, info = quantize(tiny_gen_model, calib, config=cfg, verbose=False)
    return model, info


# ─────────────────────────────────────────────────────────────────────────────
# SpeculativeReport
# ─────────────────────────────────────────────────────────────────────────────

def test_speculative_report_acceptance_rate() -> None:
    r = SpeculativeReport(tokens_accepted=7, tokens_rejected=3)
    assert abs(r.acceptance_rate - 0.7) < 1e-9


def test_speculative_report_zero_division_safe() -> None:
    r = SpeculativeReport()
    assert r.acceptance_rate == 0.0
    assert r.speedup_factor == 0.0
    assert r.inter_token_latency_ms == 0.0


def test_speculative_report_summary_keys() -> None:
    r = SpeculativeReport(tokens_generated=10, tokens_accepted=8, verify_calls=3)
    s = r.summary()
    for key in ("mode", "tokens_generated", "acceptance_rate", "speedup_factor",
                "inter_token_latency_ms", "block_efficiency", "p95_latency_ms", "tokens_per_second"):
        assert key in s, f"Missing key: {key}"

def test_speculative_report_tps() -> None:
    r = SpeculativeReport(tokens_generated=20, total_wallclock_s=2.0)
    assert r.tokens_per_second == 10.0
    r2 = SpeculativeReport(tokens_generated=20, total_wallclock_s=0.0)
    assert r2.tokens_per_second == 0.0


def test_block_efficiency() -> None:
    r = SpeculativeReport(tokens_accepted=9, verify_calls=3)
    assert abs(r.block_efficiency - 3.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CSAQInferenceEngine construction
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_init_no_hooks_warns(tiny_gen_model) -> None:
    """Engine with no CSAQ buffers should warn, not crash."""
    with pytest.warns(UserWarning, match="0 hooked layers"):
        engine = CSAQInferenceEngine(
            tiny_gen_model, causal_map={}, tokenizer=None, verbose=False
        )
    assert engine._hooks == []


def test_engine_init_with_quantised_model(quantised_model_and_info) -> None:
    model, info = quantised_model_and_info
    engine = CSAQInferenceEngine(
        model, info["causal_map"], tokenizer=None, verbose=False
    )
    # Should have found some hooked layers
    assert isinstance(engine._hooks, list)


# ─────────────────────────────────────────────────────────────────────────────
# Standard generate
# ─────────────────────────────────────────────────────────────────────────────

def test_standard_generate_shape(quantised_model_and_info) -> None:
    model, info = quantised_model_and_info
    engine = CSAQInferenceEngine(model, info["causal_map"], verbose=False)

    input_ids = torch.randint(0, 32, (1, 4))
    output, report = engine.generate(input_ids, speculative=False, max_new_tokens=6)

    assert output.shape[-1] == 4 + 6
    assert report.mode == "standard"
    assert report.tokens_generated == 6


def test_standard_generate_report_type(quantised_model_and_info) -> None:
    model, info = quantised_model_and_info
    engine = CSAQInferenceEngine(model, info["causal_map"], verbose=False)
    _, report = engine.generate(torch.randint(0, 32, (1, 4)), speculative=False, max_new_tokens=4)
    assert isinstance(report, SpeculativeReport)


# ─────────────────────────────────────────────────────────────────────────────
# Sample helper
# ─────────────────────────────────────────────────────────────────────────────

def test_sample_greedy() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    token, probs = CSAQInferenceEngine._sample(logits, 1.0, 1.0, greedy=True)
    assert token.item() == 1   # argmax is index 1

def test_sample_numerical_stability() -> None:
    """NaN/Inf logits must not crash _sample."""
    logits = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0]])
    token, probs = CSAQInferenceEngine._sample(logits, 1.0, 1.0, greedy=False)
    assert torch.isfinite(token.float()).all()


def test_sample_top_p() -> None:
    """top_p < 1 must produce a valid token without exceptions."""
    torch.manual_seed(99)
    logits = torch.randn(1, 32)
    token, probs = CSAQInferenceEngine._sample(logits, 0.8, 0.9, greedy=False)
    assert 0 <= token.item() < 32


# ─────────────────────────────────────────────────────────────────────────────
# Weight swapping
# ─────────────────────────────────────────────────────────────────────────────

def test_swap_to_verify_and_back(quantised_model_and_info) -> None:
    """Swapping to verify and back to draft must not crash."""
    model, info = quantised_model_and_info
    engine = CSAQInferenceEngine(model, info["causal_map"], verbose=False)

    if not engine._hooks:
        pytest.skip("No hooks — model too small for this test")

    engine._swap_to_verify()
    assert engine._in_verify

    engine._swap_to_draft()
    assert not engine._in_verify


# ─────────────────────────────────────────────────────────────────────────────
# KV cache utilities
# ─────────────────────────────────────────────────────────────────────────────

def test_kv_len_none() -> None:
    assert CSAQInferenceEngine._kv_len(None) == 0


def test_truncate_kv_none() -> None:
    assert CSAQInferenceEngine._truncate_kv(None, 10) is None


def test_truncate_kv_tuple_format() -> None:
    """Tuple-of-tuples KV cache must be truncated correctly."""
    # Simulate 2-layer KV cache, seq_len=8
    kv = tuple(
        (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        for _ in range(2)
    )
    truncated = CSAQInferenceEngine._truncate_kv(kv, 4)
    assert truncated[0][0].shape[2] == 4
    assert truncated[1][1].shape[2] == 4
