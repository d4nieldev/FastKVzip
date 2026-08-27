import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric import EdgeIndex

import eval as benchmark_eval
from attention.gate import Weight
from attention.kvcache import RetainCache
from data import DataWrapper
from graph import GraphBuilder, GraphScorer, GraphTopology
from utils import Evaluator, save_result, set_gen_length

try:
    import eval_graph
except ImportError:
    eval_graph = None


def _symbol(name):
    assert eval_graph is not None, "eval_graph.py is not implemented"
    value = getattr(eval_graph, name, None)
    assert value is not None, f"{name} is not implemented"
    return value


class BatchedChainBuilder(GraphBuilder):
    def forward(self, z):
        batch_size, token_count, _ = z.shape
        sources, targets = [], []
        for graph in range(batch_size):
            offset = graph * token_count
            source = torch.arange(token_count - 1, device=z.device) + offset
            sources.append(source)
            targets.append(source + 1)
        edges = torch.stack([torch.cat(sources), torch.cat(targets)])
        return GraphTopology(
            EdgeIndex(
                edges,
                sparse_size=(batch_size * token_count, batch_size * token_count),
            )
        )


def _scorer(*, layers=2, heads=2, hidden_dim=3, graph_dim=2, graph_batch=2):
    torch.manual_seed(913)
    gates = [
        Weight(
            layer,
            input_dim=hidden_dim,
            output_dim=2,
            nhead=heads,
            ngroup=1,
            dtype=torch.float64,
            sink=2,
        ).double()
        for layer in range(layers)
    ]
    scorer = GraphScorer(
        gates,
        SimpleNamespace(num_hidden_layers=layers, num_key_value_heads=heads),
        graph_dim=graph_dim,
        gin_depth=1,
        graph_builder=BatchedChainBuilder(),
        graph_microbatch_size=graph_batch,
    ).double()
    with torch.no_grad():
        scorer.b_proj.weight.normal_(std=0.1)
    return scorer


def _checkpoint_payload(*, layers=1, heads=1, hidden_dim=3, graph_dim=2):
    scorer = _scorer(
        layers=layers,
        heads=heads,
        hidden_dim=hidden_dim,
        graph_dim=graph_dim,
        graph_batch=heads,
    )
    full_state = scorer.state_dict()
    config = {
        "format_version": 1,
        "model_id": "tiny/model",
        "gate_dim": 2,
        "gate_sink": 2,
        "hidden_dim": hidden_dim,
        "num_layers": layers,
        "num_kv_heads": heads,
        "query_groups": 1,
        "graph_dim": graph_dim,
        "gin_depth": 1,
        "graph_microbatch_size": heads,
        "num_neighbors": 2,
        "knn_index": "ivf_flat",
        "ivf_nlist": 8,
        "ivf_nprobe": 2,
        "ivfpq_m": 1,
        "ivfpq_bits": 4,
        "training_mode": "two_phase",
        "token_microbatch_size": 2,
        "gate_lr": 1e-4,
        "graph_lr": 1e-3,
        "gate_lr_scheduler": None,
        "graph_lr_scheduler": None,
        "b_init": "zero",
        "freeze_gate": False,
    }
    return {
        "graph": {
            name: value.detach().cpu().clone()
            for name, value in full_state.items()
            if not name.startswith("gates.")
        },
        "gate": {
            name: value.detach().cpu().clone()
            for name, value in scorer.gates.state_dict().items()
        },
        "config": config,
        "model_id": "tiny/model",
        "prefix_ids": torch.tensor([[90, 91]], dtype=torch.long),
        "prefill_chunk": 4,
    }


def _model_config(*, layers=1, heads=1, hidden_dim=3, groups=1):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads * groups,
        hidden_size=hidden_dim,
    )


def test_eval_module_reuses_existing_benchmark_helpers():
    assert _symbol("get_data_list") is benchmark_eval.get_data_list
    assert _symbol("set_ratios") is benchmark_eval.set_ratios
    assert _symbol("DataWrapper") is DataWrapper
    assert _symbol("Evaluator") is Evaluator
    assert _symbol("set_gen_length") is set_gen_length
    assert _symbol("save_result") is save_result


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("", "_graph"),
        ("trial", "_graph_trial"),
        ("_trial", "_graph_trial"),
        ("_graph", "_graph"),
        ("_graph_trial", "_graph_trial"),
    ],
)
def test_graph_result_tags_cannot_collide_with_baseline_results(tag, expected):
    parser_default = _symbol("build_parser")().parse_args(
        ["--graph-checkpoint", "checkpoint.pt"]
    )
    assert parser_default.tag == "_graph"
    assert _symbol("_normalize_graph_tag")(tag) == expected


def test_checkpoint_is_validated_and_reconstructed_with_saved_projection_dtype(
    tmp_path,
):
    payload = _checkpoint_payload()
    path = tmp_path / "graph.pt"
    torch.save(payload, path)

    checkpoint = _symbol("load_evaluation_checkpoint")(path)
    model = SimpleNamespace(
        config=_model_config(),
        device=torch.device("cpu"),
        gates=None,
    )
    scorer = _symbol("reconstruct_graph_scorer")(checkpoint, model)

    assert checkpoint.model_id == "tiny/model"
    assert checkpoint.prefill_chunk == 4
    assert checkpoint.token_microbatch_size == 2
    assert checkpoint.graph_microbatch_size == 1
    assert scorer.a_proj.weight.dtype == torch.float64
    assert scorer.graph_builder.index_mode == "ivf_flat"
    assert scorer.graph_builder.k == 2
    expected = dict(payload["graph"])
    expected.update({f"gates.{name}": value for name, value in payload["gate"].items()})
    actual = scorer.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name].cpu(), expected[name])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload["config"].pop("graph_dim"), "graph_dim"),
        (lambda payload: payload.update(model_id="other/model"), "model identifier"),
        (
            lambda payload: payload["graph"].update(
                {"a_proj.weight": payload["graph"]["a_proj.weight"].float()}
            ),
            "projection dtype",
        ),
        (lambda payload: payload["gate"].pop("0.q_norm.weight"), "q_norm"),
    ],
)
def test_checkpoint_metadata_and_state_are_strict(tmp_path, mutation, match):
    payload = _checkpoint_payload()
    mutation(payload)
    path = tmp_path / "bad.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        _symbol("load_evaluation_checkpoint")(path)


def test_model_override_conflict_fails_before_model_loading(tmp_path):
    path = tmp_path / "graph.pt"
    torch.save(_checkpoint_payload(), path)
    calls = []

    args = SimpleNamespace(graph_checkpoint=path, model="wrong/model")
    with pytest.raises(ValueError, match="--model"):
        _symbol("run_evaluation")(args, model_factory=lambda *a, **k: calls.append(1))

    assert calls == []


def test_runtime_loads_retain_model_with_no_builtin_gate_and_validates_dimensions(
    tmp_path,
):
    path = tmp_path / "graph.pt"
    torch.save(_checkpoint_payload(), path)
    checkpoint = _symbol("load_evaluation_checkpoint")(path)
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            config=_model_config(),
            device=torch.device("cpu"),
            gates=None,
        )

    model, scorer = _symbol("build_evaluation_runtime")(
        checkpoint, model_factory=factory
    )

    assert model.gates is None
    assert scorer.training is False
    assert calls == [
        (("tiny/model",), {"kv_type": "retain", "gate_path_or_name": ""})
    ]

    bad_model = SimpleNamespace(
        config=_model_config(hidden_dim=4),
        device=torch.device("cpu"),
        gates=None,
    )
    with pytest.raises(ValueError, match="hidden size"):
        _symbol("reconstruct_graph_scorer")(checkpoint, bad_model)


def test_staged_scoring_matches_whole_context_and_bounds_hidden_token_slices(
    monkeypatch,
):
    scorer = _scorer()
    prefix = torch.full((2, 1, 3), 1000.0, dtype=torch.float64)
    context = torch.randn(2, 5, 3, dtype=torch.float64)
    hidden_cache = [
        torch.cat([prefix[layer : layer + 1], context[layer : layer + 1]], dim=1)
        for layer in range(2)
    ]
    expected = scorer(context)
    chunk_sizes = []
    propagation_shapes = []
    real_chunk = eval_graph._evaluation._hidden_chunk
    real_propagate = scorer.propagate_graph_nodes

    def tracked_chunk(*args, **kwargs):
        chunk_sizes.append(args[3] - args[2])
        return real_chunk(*args, **kwargs)

    def tracked_propagate(z, graph_ids):
        propagation_shapes.append(tuple(z.shape))
        return real_propagate(z, graph_ids)

    monkeypatch.setattr(eval_graph._evaluation, "_hidden_chunk", tracked_chunk)
    monkeypatch.setattr(scorer, "propagate_graph_nodes", tracked_propagate)

    actual = _symbol("score_hidden_cache")(
        scorer,
        hidden_cache,
        start_idx=1,
        end_idx=6,
        token_microbatch_size=2,
        graph_microbatch_size=2,
    )

    assert actual.shape == (2, 1, 2, 5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert max(chunk_sizes) == 2
    assert all(size <= 2 for size in chunk_sizes)
    assert propagation_shapes == [(2, 5, 2), (2, 5, 2)]


def test_singleton_context_dimensions_are_never_squeezed_away():
    scorer = _scorer(layers=1, heads=1, hidden_dim=1, graph_dim=1, graph_batch=1)
    hidden_cache = [torch.tensor([[[99.0], [0.25]]], dtype=torch.float64)]

    score = _symbol("score_hidden_cache")(
        scorer,
        hidden_cache,
        start_idx=1,
        end_idx=2,
        token_microbatch_size=1,
    )

    assert score.shape == (1, 1, 1, 1)


def test_hidden_layer_capture_mismatch_names_unsupported_hybrid_static_case():
    scorer = _scorer(layers=2, heads=1, graph_batch=1)

    with pytest.raises(ValueError, match="hybrid/static.*expected 2.*got 1"):
        _symbol("score_hidden_cache")(
            scorer,
            [torch.zeros(1, 3, 3, dtype=torch.float64)],
            start_idx=1,
            end_idx=3,
            token_microbatch_size=1,
        )


@pytest.mark.parametrize(
    "token_count,prefill_chunk,configured_window,expected_window",
    [(100, 200, 17, 2), (6, 6, 2, 2), (1, 8, 5, 0), (6, 6, 0, 0)],
)
def test_local_window_matches_existing_short_long_and_zero_rules(
    token_count, prefill_chunk, configured_window, expected_window
):
    original = torch.arange(token_count, dtype=torch.float64).view(1, 1, 1, -1)
    scores = original.clone()

    window = _symbol("protect_local_window")(
        scores,
        token_count=token_count,
        prefill_chunk=prefill_chunk,
        window_size=configured_window,
    )

    assert window == expected_window
    if expected_window:
        torch.testing.assert_close(
            scores[..., -expected_window:],
            torch.full_like(scores[..., -expected_window:], token_count - 1),
        )
        torch.testing.assert_close(
            scores[..., :-expected_window], original[..., :-expected_window]
        )
    else:
        torch.testing.assert_close(scores, original)


def test_context_scoring_clears_hidden_cache_on_success_and_error():
    scorer = _scorer(layers=1, heads=1, graph_batch=1)
    good = SimpleNamespace(
        start_idx=1,
        end_idx=4,
        ctx_len=3,
        device=torch.device("cpu"),
        hidden_cache=[torch.randn(1, 4, 3, dtype=torch.float64)],
        score=None,
    )

    score = _symbol("score_context_cache")(
        good,
        scorer,
        prefill_chunk=3,
        window_size=1,
        token_microbatch_size=2,
        graph_microbatch_size=1,
    )

    assert good.hidden_cache == []
    assert good.score is score
    assert score.shape == (1, 1, 1, 3)

    bad = SimpleNamespace(
        start_idx=1,
        end_idx=4,
        ctx_len=3,
        device=torch.device("cpu"),
        hidden_cache=[torch.randn(4, 3, dtype=torch.float64)],
        score=None,
    )
    with pytest.raises(ValueError, match=r"\[1,prefix\+tokens,hidden_dim\]"):
        _symbol("score_context_cache")(
            bad,
            scorer,
            prefill_chunk=4,
            window_size=1,
            token_microbatch_size=2,
        )
    assert bad.hidden_cache == []


def test_prefix_restore_copies_checkpoint_ids_to_device_without_touching_postfix():
    postfix = torch.tensor([[7, 8]])
    model = SimpleNamespace(
        device=torch.device("cpu"),
        sys_prompt_ids=torch.tensor([[999]]),
        postfix_ids=postfix,
    )
    prefix = torch.tensor([[1, 2, 3]])

    _symbol("restore_checkpoint_prefix")(model, prefix)

    torch.testing.assert_close(model.sys_prompt_ids, prefix)
    assert model.sys_prompt_ids.data_ptr() != prefix.data_ptr()
    assert model.postfix_ids is postfix


def test_existing_cache_padding_protects_prefix_and_later_query_tokens():
    cache = object.__new__(RetainCache)
    cache.sink = 2
    cache.valid = torch.tensor([[[False, True, False]]])

    padded = cache._get_valid(layer_idx=0, n_seq=7)

    assert padded.tolist() == [[True, True, False, True, False, True, True]]


def test_evaluation_restores_prefix_prefills_exact_checkpoint_and_clears_hidden_before_answer(
    tmp_path,
):
    path = tmp_path / "graph.pt"
    torch.save(_checkpoint_payload(), path)
    events = []

    class FakeModel:
        def __init__(self):
            self.config = _model_config()
            self.device = torch.device("cpu")
            self.gates = None
            self.name = "tiny-model"
            self.tokenizer = object()
            self.sys_prompt_ids = torch.tensor([[0]])
            self.postfix_ids = torch.tensor([[8]])
            self.gen_kwargs = {}

    class FakeKV:
        def __init__(self):
            self.start_idx = 2
            self.end_idx = 5
            self.ctx_len = 3
            self.device = torch.device("cpu")
            self.hidden_cache = [torch.randn(1, 5, 3, dtype=torch.float64)]
            self.score = None

        def prune(self, ratio, level):
            assert self.score.shape == (1, 1, 1, 3)
            return 0.25, ratio

    class FakeWrapper:
        def __init__(self, name, dataset, model):
            self.model = model
            self.model.sys_prompt_ids = torch.tensor([[999]])
            self.model.postfix_ids = torch.tensor([[44]])

        def __len__(self):
            return 1

        def prefill_context(self, index, **kwargs):
            events.append(("prefill", self.model.sys_prompt_ids.clone(), kwargs))
            return FakeKV()

        def generate_answer(self, index, kv, prob):
            assert kv.hidden_cache == []
            events.append(("answer", self.model.postfix_ids.clone(), prob))
            return {"qa": {"q": None, "a": None, "gt": None}}, {"qa": {}}

    class FakeEvaluator:
        def __init__(self, model, inputs, info):
            pass

        def __call__(self, kv, generate):
            return {"qa": {"pruned": "p", "full__": "f", "answer": "a"}}

    class FakeTimestamp:
        def __init__(self, verbose):
            pass

        def __call__(self, message):
            events.append(("time", message))

    saved = []
    args = SimpleNamespace(
        graph_checkpoint=path,
        model=None,
        data="squad",
        idx=0,
        num=1,
        tag="",
        window_size=1,
        level="pair",
    )

    _symbol("run_evaluation")(
        args,
        model_factory=lambda *args, **kwargs: FakeModel(),
        dataset_loader=lambda name, tokenizer: [object()],
        wrapper_factory=FakeWrapper,
        evaluator_factory=FakeEvaluator,
        result_saver=lambda *values: saved.append(values),
        generation_length_setter=lambda *args: None,
        timestamp_factory=FakeTimestamp,
    )

    prefill = events[0]
    assert prefill[0] == "prefill"
    torch.testing.assert_close(prefill[1], torch.tensor([[90, 91]]))
    assert prefill[2] == {
        "prefill_chunk": 4,
        "save_hidden": True,
        "do_score": False,
    }
    assert events[1] == ("answer", torch.tensor([[44]]), False)
    assert len(saved) == 1
