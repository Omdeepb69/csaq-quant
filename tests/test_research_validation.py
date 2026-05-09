import json
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from benchmarks.research_validation import (
    measure_memory_gb,
    run_section_a,
    run_section_b
)
from csaq.config import CSAQConfig

class _TinyModel(nn.Module):
    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, vocab)
        self.config = type("cfg", (), {
            "eos_token_id": 1,
            "tie_word_embeddings": False,
            "model_type": "tiny"
        })()
        
    def forward(self, input_ids, labels=None, **kw):
        x = self.embed(input_ids)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab), labels.view(-1))
            
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        return out
        
    def get_input_embeddings(self):
        return self.embed
        
    def get_output_embeddings(self):
        return self.fc
        
    def parameters(self, recurse=True):
        return super().parameters(recurse)

class _DummyArgs:
    def __init__(self, output_dir):
        self.model_path = "dummy"
        self.domain_name = "konkani"
        self.target_bits = 4.0
        self.bit_options = "4,8"
        self.seq_len = 8
        self.output_dir = output_dir

def _mock_get_fresh_model(model_path, device_map, dtype):
    torch.manual_seed(42)
    return _TinyModel()

def test_measure_memory_returns_positive_float():
    mem = measure_memory_gb()
    assert isinstance(mem, float)
    assert mem > 0

def test_section_a_runs_on_tiny_model(monkeypatch):
    import benchmarks.research_validation as rv
    monkeypatch.setattr(rv, "get_fresh_model", _mock_get_fresh_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _DummyArgs(tmpdir)
        tokenizer = None
        
        calib_data = []
        for _ in range(8):
            ids = torch.randint(0, 32, (1, 8))
            calib_data.append({"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)})
            
        eval_data = ["This is a test."] * 8
        
        # mock compute_perplexity because we have no real tokenizer
        monkeypatch.setattr(rv, "compute_perplexity", lambda m, t, d, seq_len: 15.5)
        
        results = {}
        json_path = os.path.join(tmpdir, "results.json")
        
        run_section_a(args, tokenizer, "cpu", torch.float32, calib_data, calib_data, eval_data, eval_data, results, json_path)
        
        assert "section_a" in results
        assert results["section_a"]["fp32_en"] == 15.5
        assert "gain_pts" in results["section_a"]
        assert os.path.exists(json_path)
        assert os.path.exists(os.path.join(tmpdir, "ppl_benchmark.csv"))

def test_section_b_ablation_produces_two_ppl_values(monkeypatch):
    import benchmarks.research_validation as rv
    monkeypatch.setattr(rv, "get_fresh_model", _mock_get_fresh_model)
    monkeypatch.setattr(rv, "compute_perplexity", lambda m, t, d, seq_len: 20.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _DummyArgs(tmpdir)
        tokenizer = None
        
        calib_data = []
        for _ in range(8):
            ids = torch.randint(0, 32, (1, 8))
            calib_data.append({"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)})
            
        eval_data = ["This is a test."] * 8
        results = {}
        json_path = os.path.join(tmpdir, "results.json")
        os.makedirs(os.path.join(tmpdir, "figures"), exist_ok=True)
        
        run_section_b(args, tokenizer, "cpu", torch.float32, calib_data, calib_data, eval_data, results, json_path)
        
        assert "section_b" in results
        assert "jaccard_ppl" in results["section_b"]
        assert "perchan_ppl" in results["section_b"]

def test_konkani_sample_file_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "konkani_sample.txt")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) >= 64

def test_results_json_written_after_section(monkeypatch):
    import benchmarks.research_validation as rv
    monkeypatch.setattr(rv, "get_fresh_model", _mock_get_fresh_model)
    monkeypatch.setattr(rv, "compute_perplexity", lambda m, t, d, seq_len: 10.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _DummyArgs(tmpdir)
        calib_data = [{"input_ids": torch.randint(0, 32, (1, 8)), "attention_mask": torch.ones((1,8))}] * 2
        results = {}
        json_path = os.path.join(tmpdir, "out.json")
        
        run_section_a(args, None, "cpu", torch.float32, calib_data, calib_data, ["test"], ["test"], results, json_path)
        
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "section_a" in data
