import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graph import ImplicitGraphScorer, save_checkpoint
from graph.evaluation import (
    _clear_hidden_cache,
    load_evaluation_checkpoint,
    protect_local_window,
    score_context_cache,
    score_hidden_cache,
)


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead, self.ngroup, self.output_dim, self.sink, self.d = 1, 1, 1, 1, 1.0
        self.q_proj = nn.Linear(2, 1)
        self.k_proj = nn.Linear(2, 1, bias=False)
        self.q_norm = _ScaleNorm(1)
        self.k_norm = _ScaleNorm(1)
        self.k_base = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1))


class _ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        return value * self.weight


def _config():
    return SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=2,
    )


def _scorer():
    return ImplicitGraphScorer(
        [Gate().double()],
        _config(),
        graph_dim=2,
        graph_microbatch_size=1,
        compute_dtype=torch.float64,
    )


def _checkpoint_config():
    return {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": 1,
        "gate_sink": 1,
        "hidden_dim": 2,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 2,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 2,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
    }


def test_score_hidden_cache_matches_full_scorer_across_token_chunks():
    torch.manual_seed(8)
    scorer = _scorer()
    context = torch.randn(5, 2, dtype=torch.float64)
    hidden_cache = [torch.cat((torch.zeros(1, 2, 2, dtype=torch.float64), context.unsqueeze(0)), dim=1)]
    actual = score_hidden_cache(
        scorer, hidden_cache, start_idx=2, end_idx=7, token_microbatch_size=2
    )
    expected = scorer(context.unsqueeze(0), token_microbatch_size=5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_current_checkpoint_validation_has_only_implicit_mixer_state(tmp_path):
    scorer = _scorer()
    path = save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config=_checkpoint_config(),
        model_id="unit",
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor={"epoch": 0},
        wandb_run_id=None,
    )
    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.config["gram_normalization"] == "token-count"
    payload = torch.load(path, weights_only=False)
    assert set(payload["mixer"]) == {
        "mixer.in_proj.weight",
        "mixer.out_proj.weight",
        "mixer.gamma",
        "mixer.beta",
        "mixer.alpha",
    }
    payload["mixer"].pop("mixer.alpha")
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="missing tensor"):
        load_evaluation_checkpoint(bad)
    payload = torch.load(path, weights_only=False)
    payload["config"]["activation_order"] = "leaky-relu-batchnorm"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="activation order"):
        load_evaluation_checkpoint(bad)


def test_protection_only_changes_the_context_local_window_and_hidden_cache_is_cleared():
    scores = torch.arange(5.0).view(1, 1, 1, 5)
    window = protect_local_window(scores, token_count=5, prefill_chunk=4, window_size=2)
    assert window == 2
    assert scores[..., :3].tolist() == [[[[0.0, 1.0, 2.0]]]]
    assert scores[..., -2:].eq(scores.max()).all()
    kv = SimpleNamespace(hidden_cache=[torch.zeros(1)])
    _clear_hidden_cache(kv)
    assert kv.hidden_cache == []


def test_score_context_cache_preserves_non_context_indices_and_releases_hidden():
    scorer = _scorer()
    kv = SimpleNamespace(
        start_idx=2,
        end_idx=5,
        ctx_len=3,
        device=scorer.device,
        hidden_cache=[torch.randn(1, 5, 2, dtype=torch.float64)],
        score=None,
    )
    scores = score_context_cache(
        kv, scorer, prefill_chunk=4, window_size=1, token_microbatch_size=2
    )
    assert scores.shape == (1, 1, 1, 3)
    assert kv.hidden_cache == []
