from types import SimpleNamespace

import pytest
import torch

import graph.evaluation as graph_evaluation
from graph.evaluation import score_context_chunk_cache
from model import ModelKVzip


def _use_hidden_values_as_scores(monkeypatch):
    def score(_scorer, hidden_cache, *, start_idx, end_idx, **_kwargs):
        values = hidden_cache[0][:, start_idx:end_idx, 0]
        return values.view(1, 1, 1, -1)

    monkeypatch.setattr(graph_evaluation, "score_hidden_cache", score)


def test_each_prefill_chunk_is_scored_as_an_independent_graph(monkeypatch):
    _use_hidden_values_as_scores(monkeypatch)
    first_hidden = [torch.arange(10.0).view(1, 5, 2)]
    kv = SimpleNamespace(
        start_idx=2,
        hidden_cache=first_hidden,
        score=[torch.zeros(1, 1, 0)],
    )

    first = score_context_chunk_cache(kv, object(), token_microbatch_size=2)
    first_scores = kv.score[0].clone()

    assert first.tolist() == [[[[4.0, 6.0, 8.0]]]]
    assert kv.score[0].tolist() == [[[0.0, 0.0, 4.0, 6.0, 8.0]]]
    assert kv.hidden_cache == []

    kv.hidden_cache = [torch.tensor([[[20.0, 0.0], [22.0, 0.0]]])]
    second = score_context_chunk_cache(kv, object(), token_microbatch_size=3)

    torch.testing.assert_close(kv.score[0][..., :5], first_scores)
    assert second.tolist() == [[[[20.0, 22.0]]]]
    assert kv.score[0][..., 5:].tolist() == [[[20.0, 22.0]]]
    assert kv.hidden_cache == []


def test_chunk_callback_runs_before_official_pruning(monkeypatch):
    _use_hidden_values_as_scores(monkeypatch)
    events = []

    class Cache:
        def __init__(self, evict_range):
            self.start_idx, self.end_idx = evict_range
            self.ctx_len = self.end_idx - self.start_idx
            self.hidden_cache = []
            self.valid = None

        def init_score(self, get_score=False):
            self.compute_gate = not get_score
            self.score = [torch.zeros(1, 1, 0)]

        def prune_chunk(self, ratio, evict_range, level):
            events.append(("prune", evict_range, ratio, level))
            valid = torch.ones(1, 1, evict_range[1] - evict_range[0], dtype=torch.bool)
            self.valid = valid if self.valid is None else torch.cat((self.valid, valid), -1)
            return 0.0, ratio

    class Model(ModelKVzip):
        def __init__(self):
            self.sys_prompt_ids = torch.tensor([[90, 91]])
            self.gates = None
            self.kv_type = "retain"

        def _init_kv(self, kv=None, evict_range=(0, 0)):
            return Cache(evict_range)

        def __call__(self, input_ids, kv, update_cache=False, **_kwargs):
            kv.hidden_cache.append(torch.ones(1, input_ids.size(1), 2))
            events.append(("forward", input_ids.size(1)))

    def score_chunk(kv):
        events.append(("score", kv.hidden_cache[0].size(1)))
        score_context_chunk_cache(kv, object(), token_microbatch_size=6)

    kv = Model().prefill(
        torch.arange(8).view(1, -1),
        prefill_chunk_size=6,
        window_size=0.75,
        chunk_ratio=0.9,
        level="pair",
        chunk_scorer=score_chunk,
    )

    assert events == [
        ("forward", 6),
        ("score", 6),
        ("forward", 4),
        ("score", 4),
        ("prune", (2, 4), pytest.approx(0.6), "pair"),
    ]
    assert kv.valid.shape[-1] == kv.ctx_len == 8
    assert kv.hidden_cache == []
