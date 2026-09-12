import copy
import importlib
import math
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from graph import ImplicitGraphScorer


def _primitive(name):
    try:
        module = importlib.import_module("graph.answer_training")
    except ModuleNotFoundError:
        pytest.fail("graph.answer_training is missing")
    try:
        return getattr(module, name)
    except AttributeError:
        pytest.fail(f"graph.answer_training.{name} is missing")


class _ScaleNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, value):
        return value * self.weight


class _Gate(nn.Module):
    def __init__(self, heads=1, *, trainable_norm=False):
        super().__init__()
        self.nhead = heads
        self.ngroup = self.output_dim = self.sink = 1
        self.d = 1.0
        self.q_proj = nn.Linear(2, heads, bias=True, dtype=torch.float64)
        self.k_proj = nn.Linear(2, heads, bias=False, dtype=torch.float64)
        self.q_norm = _ScaleNorm() if trainable_norm else nn.Identity()
        self.k_norm = _ScaleNorm() if trainable_norm else nn.Identity()
        self.k_base = nn.Parameter(torch.ones(heads, 1, 1, 1, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(heads, 1, 1, dtype=torch.float64))


def _scorer(layers=1, heads=1, *, trainable_norm=False):
    config = SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads,
        hidden_size=2,
    )
    return ImplicitGraphScorer(
        [_Gate(heads, trainable_norm=trainable_norm) for _ in range(layers)],
        config,
        graph_dim=2,
        graph_microbatch_size=heads,
        compute_dtype=torch.float64,
    )


def _serial_subgraph_scores(scorer, hidden_by_layer, subgraph_size):
    """Independent oracle: each scorer forward sees exactly one subgraph."""
    hidden = torch.stack([value[0] if value.ndim == 3 else value for value in hidden_by_layer])
    return torch.cat([
        scorer(hidden[:, start:start + subgraph_size], token_microbatch_size=subgraph_size)
        for start in range(0, hidden.size(1), subgraph_size)
    ], dim=-1)


def test_uniform_retention_uses_explicit_resumable_rng_state():
    retention_ratio = _primitive("retention_ratio")
    rng = random.Random(17)
    prefix = [
        retention_ratio("uniform", 0.2, 0.8, global_step=step, total_steps=8, rng=rng)
        for step in range(3)
    ]
    state = rng.getstate()
    suffix = [
        retention_ratio("uniform", 0.2, 0.8, global_step=step, total_steps=8, rng=rng)
        for step in range(3, 8)
    ]
    resumed = random.Random()
    resumed.setstate(state)
    replayed = [
        retention_ratio(
            "uniform", 0.2, 0.8, global_step=step, total_steps=8, rng=resumed
        )
        for step in range(3, 8)
    ]

    assert suffix == replayed
    assert all(0.2 <= ratio <= 0.8 for ratio in prefix + suffix)
    with pytest.raises(ValueError, match="RNG"):
        retention_ratio("uniform", 0.2, 0.8, global_step=0, total_steps=8)


def test_linear_retention_decays_over_flattened_global_horizon():
    retention_ratio = _primitive("retention_ratio")
    actual = [
        retention_ratio("linear", 0.2, 0.8, global_step=step, total_steps=5)
        for step in range(5)
    ]

    assert actual == pytest.approx([0.8, 0.65, 0.5, 0.35, 0.2])
    assert actual == sorted(actual, reverse=True)
    assert retention_ratio(
        "linear", 0.2, 0.8, global_step=-10, total_steps=5
    ) == pytest.approx(0.8)
    assert retention_ratio(
        "linear", 0.2, 0.8, global_step=99, total_steps=5
    ) == pytest.approx(0.2)
    assert retention_ratio(
        "linear", 0.2, 0.8, global_step=0, total_steps=1
    ) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="uniform or linear"):
        retention_ratio(
            "global-linear", 0.2, 0.8, global_step=0, total_steps=5
        )


def test_retained_token_count_uses_floor_with_a_one_token_minimum():
    retained_token_count = _primitive("retained_token_count")

    assert retained_token_count(9, 0.33) == 2
    assert retained_token_count(3, 0.1) == 1
    assert retained_token_count(3, 0.0) == 1
    assert retained_token_count(3, 1.0) == 3
    with pytest.raises(ValueError, match="positive"):
        retained_token_count(0, 0.5)


def test_fallback_validation_reserves_last_ceil_ten_percent_after_deduplication():
    fallback_validation_split = _primitive("fallback_validation_split")
    requested = ["a", "b", "a", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    training, validation = fallback_validation_split(requested)

    assert training == ("a", "b", "c", "d", "e", "f", "g", "h", "i")
    assert validation == ("j", "k")
    with pytest.raises(ValueError, match="at least two"):
        fallback_validation_split(["only", "only"])


def test_context_scores_are_selected_once_globally(monkeypatch):
    global_topk_indices = _primitive("global_topk_indices")
    scores = torch.tensor([[[[9.0, 1.0, 8.0, 0.0], [0.0, 7.0, 6.0, 10.0]]]])
    topk_calls = []
    original_topk = torch.topk

    def record_topk(input, k, **kwargs):
        topk_calls.append((input.detach().clone(), k, kwargs))
        return original_topk(input, k, **kwargs)

    monkeypatch.setattr(torch, "topk", record_topk)
    indices = global_topk_indices(scores, 0.25)

    assert indices.tolist() == [[[[0], [3]]]]
    assert len(topk_calls) == 1
    torch.testing.assert_close(topk_calls[0][0], scores)


@pytest.mark.parametrize("tokens,token_budget", [(2, 24), (3, 3), (24, 24), (32, 3), (32, 6), (32, 24)])
def test_grouped_scores_match_serial_subgraphs_and_use_the_token_budget(monkeypatch, tokens, token_budget):
    torch.manual_seed(18)
    scorer = _scorer(layers=2, heads=3, trainable_norm=True)
    hidden = [torch.randn(tokens, 2, dtype=torch.float64) for _ in range(2)]
    expected = _serial_subgraph_scores(scorer, hidden, 3)
    packed_shapes = []
    prepare = scorer.prepare

    def observe_prepare(values, graph_ids, **kwargs):
        packed_shapes.append(tuple(values.shape))
        return prepare(values, graph_ids, **kwargs)

    monkeypatch.setattr(scorer, "prepare", observe_prepare)
    actual = _primitive("score_context_subgraphs")(
        scorer, hidden, subgraph_size=3,
        token_microbatch_size=token_budget, graph_microbatch_size=4,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    full_count, tail = divmod(tokens, 3)
    group_sizes = [min(token_budget // 3, full_count - start)
                   for start in range(0, full_count, token_budget // 3)]
    expected_shapes = [(graphs * count, 3, 2) for graphs in (4, 2) for count in group_sizes]
    if tail:
        expected_shapes.extend((graphs, tail, 2) for graphs in (4, 2))
    assert sorted(packed_shapes) == sorted(expected_shapes)
    assert max(graphs * length for graphs, length, _dim in packed_shapes) <= 4 * token_budget
    if tokens >= 6:
        changed = [values.clone() for values in hidden]
        for values in changed:
            values[3:6] += 100
        changed_scores = _primitive("score_context_subgraphs")(
            scorer, changed, subgraph_size=3,
            token_microbatch_size=token_budget, graph_microbatch_size=4,
        )
        torch.testing.assert_close(changed_scores[..., :3], actual[..., :3])
        torch.testing.assert_close(changed_scores[..., 6:], actual[..., 6:])


def test_compaction_preserves_outside_tokens_and_has_exact_hard_forward_with_value_only_ste():
    compact_context_kv = _primitive("compact_context_kv")
    score_gradient_health = _primitive("score_gradient_health")

    keys = [torch.arange(6.0, dtype=torch.float64).view(1, 1, 6, 1)]
    values = [
        torch.tensor(
            [[[[10.0], [1.0], [2.0], [3.0], [4.0], [20.0]]]],
            dtype=torch.float64,
        )
    ]
    raw_scores = torch.tensor(
        [[[[0.0, 5.0, 1.0, 4.0]]]], dtype=torch.float64, requires_grad=True
    )
    compacted = compact_context_kv(
        keys,
        values,
        raw_scores,
        ratio=0.5,
        temperature=0.7,
        context_range=(1, 5),
    )
    expected_positions = torch.tensor([0, 2, 4, 5])

    assert compacted.indices.tolist() == [[[[1, 3]]]]
    assert torch.equal(compacted.keys[0], keys[0].index_select(2, expected_positions))
    assert torch.equal(
        compacted.values[0].detach(), values[0].index_select(2, expected_positions)
    )
    assert compacted.keys[0].grad_fn is None

    compacted.values[0][:, :, 1:3].square().sum().backward()
    health = score_gradient_health(raw_scores.grad, compacted.indices)

    assert raw_scores.grad is not None
    assert raw_scores.grad[..., [1, 3]].abs().min() > 0
    assert raw_scores.grad[..., [0, 2]].abs().min() > 0
    assert health.overall > 0
    assert health.retained > 0
    assert health.evicted > 0


def test_validation_compaction_is_hard_and_builds_no_softmax_graph():
    compact_context_kv = _primitive("compact_context_kv")
    keys = [torch.arange(4.0).view(1, 1, 4, 1)]
    values = [torch.arange(4.0).view(1, 1, 4, 1)]
    raw_scores = torch.tensor([[[[1.0, 4.0, 3.0, 2.0]]]], requires_grad=True)

    compacted = compact_context_kv(
        keys,
        values,
        raw_scores,
        ratio=0.5,
        temperature=1.0,
        context_range=(0, 4),
        straight_through=False,
    )

    assert compacted.indices.tolist() == [[[[1, 2]]]]
    assert compacted.values[0].grad_fn is None


def test_compaction_copies_only_retained_data_from_inference_cache_tensors(monkeypatch):
    compact_context_kv = _primitive("compact_context_kv")
    inference_clone_sizes = []
    original_clone = torch.Tensor.clone

    def record_clone(tensor, *args, **kwargs):
        if tensor.is_inference():
            inference_clone_sizes.append(tensor.numel())
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", record_clone)
    with torch.inference_mode():
        keys = [torch.arange(64.0).view(1, 1, 64, 1)]
        values = [torch.arange(64.0).view(1, 1, 64, 1)]
    raw_scores = torch.tensor(
        [[[[float(i) for i in range(62)]]]], requires_grad=True
    )

    compacted = compact_context_kv(
        keys,
        values,
        raw_scores,
        ratio=0.1,
        temperature=1.0,
        context_range=(1, 63),
    )
    compacted.values[0].sum().backward()

    assert compacted.values[0].numel() == 8
    assert max(inference_clone_sizes, default=0) <= compacted.values[0].numel()
    assert not compacted.keys[0].is_inference()
    assert not compacted.values[0].is_inference()
    assert raw_scores.grad is not None


def test_answer_objective_masks_non_answer_tokens_and_reports_token_weighted_sums():
    answer_objective = _primitive("answer_objective")
    token_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros(1, 4, 4, dtype=torch.float64)
    logits[0, 0, 3] = 20.0  # ignored: predicts the query token at position 1
    logits[0, 1] = torch.tensor([0.0, 0.0, 4.0, 0.0])  # answer token 2: correct
    logits[0, 2] = torch.tensor([5.0, 0.0, 0.0, 1.0])  # answer token 3: wrong
    expected_sum = F.cross_entropy(
        logits[:, 1:3].reshape(-1, 4), token_ids[:, 2:4].reshape(-1), reduction="sum"
    )

    objective = answer_objective(logits, token_ids, answer_start=2)

    torch.testing.assert_close(objective.nll_sum, expected_sum)
    torch.testing.assert_close(objective.loss, expected_sum / 2)
    assert objective.correct_tokens == 1
    assert objective.token_count == 2
    assert objective.accuracy == pytest.approx(0.5)


@pytest.mark.parametrize("token_budget", [3, 24])
@pytest.mark.parametrize("graph_budget", [1, 4])
def test_score_replay_memory_is_bounded_by_both_microbatches(monkeypatch, token_budget, graph_budget):
    replay_score_gradients = _primitive("replay_score_gradients")
    inference_clone_shapes = []
    original_clone = torch.Tensor.clone

    def record_clone(tensor, *args, **kwargs):
        if tensor.is_inference():
            inference_clone_shapes.append(tuple(tensor.shape))
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", record_clone)

    def peak_saved_elements(token_count):
        live = peak = 0

        class SavedTensor:
            def __init__(self, tensor):
                nonlocal live, peak
                self.tensor = tensor
                live += tensor.numel()
                peak = max(peak, live)

            def __del__(self):
                nonlocal live
                live -= self.tensor.numel()

        with torch.inference_mode():
            hidden = [torch.randn(1, token_count, 2, dtype=torch.float64) for _ in range(2)]
        scorer = _scorer(layers=2, heads=3, trainable_norm=True)
        score_batch = scorer.score_subgraph_batch

        def observe_batch(*args, **kwargs):
            assert live == 0, "Replay must release the previous token/graph batch before scoring another"
            return score_batch(*args, **kwargs)

        monkeypatch.setattr(scorer, "score_subgraph_batch", observe_batch)
        with torch.autograd.graph.saved_tensors_hooks(
            SavedTensor, lambda saved: saved.tensor
        ):
            replay_score_gradients(
                scorer,
                hidden,
                torch.ones(2, 1, 3, token_count, dtype=torch.float64),
                torch.zeros(2, 1, 3, 1, dtype=torch.long),
                subgraph_size=3,
                token_microbatch_size=token_budget,
                graph_microbatch_size=graph_budget,
            )
        assert live == 0
        return peak

    one_microbatch_peak = peak_saved_elements(token_budget)
    assert one_microbatch_peak > 0
    assert peak_saved_elements(9 * token_budget + 2) <= one_microbatch_peak
    assert all(shape[-2] <= token_budget for shape in inference_clone_shapes)


@pytest.mark.parametrize("token_budget", [3, 6, 24])
@pytest.mark.parametrize("batched_hidden", [False, True])
def test_grouped_replay_vjp_matches_serial_parameter_gradients(token_budget, batched_hidden):
    torch.manual_seed(29)
    direct = _scorer(layers=2, heads=3, trainable_norm=True)
    replayed = copy.deepcopy(direct)
    with torch.inference_mode():
        hidden = [torch.randn(32, 2, dtype=torch.float64) for _ in range(2)]
        if batched_hidden:
            hidden = [value.unsqueeze(0) for value in hidden]
    gradient = torch.randn(2, 1, 3, 32, dtype=torch.float64)
    selected = torch.tensor([0, 7, 24, 31]).view(1, 1, 1, 4).expand(2, 1, 3, 4)
    # Nonzero existing gradients catch an accidental zero_grad or averaging
    # when replay adds this question to earlier questions in an update.
    for left, right in zip(direct.parameters(), replayed.parameters()):
        left.grad = torch.randn_like(left)
        right.grad = left.grad.clone()
    _serial_subgraph_scores(direct, hidden, 3).backward(gradient)
    health = _primitive("replay_score_gradients")(
        replayed, hidden, gradient, selected, subgraph_size=3,
        token_microbatch_size=token_budget, graph_microbatch_size=4,
    )
    for (name, expected), (actual_name, actual) in zip(
        direct.named_parameters(), replayed.named_parameters()
    ):
        assert actual_name == name
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-10, atol=2e-10)
    expected_health = _primitive("score_gradient_health")(gradient, selected)
    for field in ("overall", "retained", "evicted"):
        torch.testing.assert_close(getattr(health, field), getattr(expected_health, field))


@pytest.mark.parametrize("gradient_tokens", [4, 6])
def test_score_replay_rejects_wrong_gradient_length_before_accumulating(gradient_tokens):
    scorer = _scorer()
    with pytest.raises(ValueError, match="external gradient"):
        _primitive("replay_score_gradients")(
            scorer,
            [torch.randn(5, 2, dtype=torch.float64)],
            torch.ones(1, 1, 1, gradient_tokens, dtype=torch.float64),
            torch.tensor([[[[0]]]]),
            subgraph_size=2,
            token_microbatch_size=2,
        )
    assert all(parameter.grad is None for parameter in scorer.parameters())


@pytest.mark.parametrize("subgraph_size", [None, 2])
@pytest.mark.parametrize("batched_hidden", [False, True])
def test_external_score_gradient_replay_matches_direct_chunked_scorer_gradients(
    subgraph_size, batched_hidden
):
    compact_context_kv = _primitive("compact_context_kv")
    freeze_llm = _primitive("freeze_llm")
    replay_score_gradients = _primitive("replay_score_gradients")
    score_context_subgraphs = _primitive("score_context_subgraphs")

    torch.manual_seed(23)
    direct = _scorer()
    replayed = copy.deepcopy(direct)
    with torch.inference_mode():
        hidden = [torch.randn(5, 2, dtype=torch.float64)]
        if batched_hidden:
            hidden = [value.unsqueeze(0) for value in hidden]
    keys = [torch.randn(1, 1, 5, 2, dtype=torch.float64)]
    values = [torch.randn(1, 1, 5, 2, dtype=torch.float64)]
    llm = freeze_llm(nn.Linear(2, 1, bias=False, dtype=torch.float64))
    llm_before = [parameter.detach().clone() for parameter in llm.parameters()]

    direct_scores = _serial_subgraph_scores(direct, hidden, subgraph_size or 5)
    direct_compacted = compact_context_kv(
        keys,
        values,
        direct_scores,
        ratio=0.6,
        temperature=0.9,
        context_range=(0, 5),
    )
    llm(direct_compacted.values[0].reshape(-1, 2)).square().sum().backward()

    with torch.no_grad():
        initial_scores = score_context_subgraphs(
            replayed, hidden, subgraph_size=subgraph_size, token_microbatch_size=2
        )
    score_leaf = initial_scores.detach().requires_grad_(True)
    replay_compacted = compact_context_kv(
        keys,
        values,
        score_leaf,
        ratio=0.6,
        temperature=0.9,
        context_range=(0, 5),
    )
    llm(replay_compacted.values[0].reshape(-1, 2)).square().sum().backward()
    assert all(parameter.grad is None for parameter in replayed.parameters())
    health = replay_score_gradients(
        replayed,
        hidden,
        score_leaf.grad,
        replay_compacted.indices,
        subgraph_size=subgraph_size,
        token_microbatch_size=2,
    )

    for (direct_name, direct_parameter), (replay_name, replay_parameter) in zip(
        direct.named_parameters(), replayed.named_parameters()
    ):
        assert replay_name == direct_name
        torch.testing.assert_close(
            replay_parameter.grad, direct_parameter.grad, rtol=1e-10, atol=1e-10
        )
    assert health.overall > 0 and health.retained > 0 and health.evicted > 0
    assert all(parameter.grad is None for parameter in llm.parameters())

    scorer_before = [parameter.detach().clone() for parameter in replayed.parameters()]
    torch.optim.SGD(replayed.parameters(), lr=0.01).step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(scorer_before, replayed.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(llm_before, llm.parameters())
    )


def test_freezing_llm_keeps_answer_forward_differentiable_for_values():
    freeze_llm = _primitive("freeze_llm")
    llm = freeze_llm(nn.Linear(2, 3, dtype=torch.float64))
    compact_values = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

    llm(compact_values).sum().backward()

    assert compact_values.grad is not None and compact_values.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in llm.parameters())
    assert all(parameter.grad is None for parameter in llm.parameters())


@pytest.mark.parametrize(
    "identity",
    [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        SimpleNamespace(config=SimpleNamespace(model_type="qwen3")),
        SimpleNamespace(config=SimpleNamespace(model_type="llama")),
    ],
)
def test_answer_training_accepts_qwen_and_llama_identities(identity):
    validate_answer_training_model_identity = _primitive(
        "validate_answer_training_model_identity"
    )

    assert validate_answer_training_model_identity(identity) in {"qwen", "llama"}


@pytest.mark.parametrize(
    "identity",
    [
        "google/gemma-3-4b-it",
        SimpleNamespace(config=SimpleNamespace(model_type="gemma3")),
    ],
)
def test_answer_training_rejects_gemma3_identity(identity):
    validate_answer_training_model_identity = _primitive(
        "validate_answer_training_model_identity"
    )

    with pytest.raises(ValueError, match="Gemma3.*not supported"):
        validate_answer_training_model_identity(identity)
