import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import eval_graph
from attention.score import KVScore
from data import DataWrapper
from graph import ImplicitGraphScorer, save_checkpoint
from graph.evaluation import (
    _clear_hidden_cache,
    load_evaluation_checkpoint,
    protect_local_window,
    score_context_cache,
    score_hidden_cache,
)
from utils import Evaluator


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
    assert (
        protect_local_window(
            scores, token_count=5, prefill_chunk=5, window_size=10
        )
        == 5
    )
    kv = SimpleNamespace(hidden_cache=[torch.zeros(1)])
    _clear_hidden_cache(kv)
    assert kv.hidden_cache == []


@pytest.mark.parametrize(
    "level", ("pair", "pair-head", "pair-layer", "adakv-layer")
)
def test_protected_window_survives_when_larger_than_pruning_budget(level):
    scores = torch.arange(10.0).view(1, 1, 1, 10)
    window = protect_local_window(
        scores, token_count=10, prefill_chunk=10, window_size=4
    )
    cache = KVScore()
    cache.protected_window = window

    valid, _ = cache.threshold(scores, ratio=0.2, level=level)

    assert valid[..., -window:].all()
    assert valid.float().mean().item() >= window / scores.size(-1)


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
        kv, scorer, prefill_chunk=3, window_size=1, token_microbatch_size=2
    )
    assert scores.shape == (1, 1, 1, 3)
    assert kv.protected_window == 1
    assert kv.hidden_cache == []


def test_full_cache_answer_can_be_disabled_without_skipping_pruned_generation():
    class Model:
        name = "unit"

        def __init__(self):
            self.generated = 0

        def set_chat_template(self, _task):
            pass

        def apply_template(self, _query):
            return torch.tensor([[1]])

        def generate(self, _query, *, kv):
            self.generated += 1
            return "pruned"

        def encode(self, text):
            return torch.tensor([[2 if text == "gold" else 3]])

        def decode(self, ids):
            return "gold" if ids.item() == 2 else "pruned"

    parser = eval_graph.build_parser()
    assert parser.parse_args(["--graph-checkpoint", "checkpoint.pt"]).full_cache_answer
    options = parser.parse_args(
        ["--graph-checkpoint", "checkpoint.pt", "--no-full-cache-answer"]
    )
    assert not options.full_cache_answer

    model = Model()
    dataset = DataWrapper(
        "unit",
        [{"question": ["question"], "answers": ["gold"]}],
        model,
    )
    inputs, info = dataset.generate_answer(
        0, object(), prob=False, full_cache_answer=False
    )
    assert model.generated == 0

    result = Evaluator(model, inputs, info)(object())
    assert model.generated == 1
    assert result["qa"] == {
        "pruned": "pruned",
        "full__": None,
        "answer": "gold",
    }
    with pytest.raises(ValueError, match="probabilities require"):
        dataset.generate_answer(0, object(), prob=True, full_cache_answer=False)

    model = Model()
    dataset = DataWrapper(
        "unit",
        [{"question": ["question"], "answers": ["gold"]}],
        model,
    )
    inputs, info = dataset.generate_answer(0, object(), prob=False)
    assert model.generated == 1
    result = Evaluator(model, inputs, info)(object())
    assert model.generated == 2
    assert result["qa"]["full__"] == "pruned"
