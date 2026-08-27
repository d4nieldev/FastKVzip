import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import EdgeIndex

from attention.gate import Weight
from graph import GraphBuilder, GraphScorer, GraphTopology

try:
    import graph.training as training
except ImportError:
    training = None


def _symbol(name):
    assert training is not None, "graph.training is not implemented"
    value = getattr(training, name, None)
    assert value is not None, f"{name} is not implemented"
    return value


def test_training_entrypoints_are_available_from_graph_package():
    import graph

    for name in (
        "TeacherExample",
        "GraphTrainer",
        "SchedulerSpec",
        "parse_scheduler_spec",
        "build_scheduler",
        "build_adamw_optimizers",
        "resolve_joint_settings",
        "resolve_b_init",
        "initialize_b_projection",
        "load_gate_checkpoint",
        "save_checkpoint",
        "load_checkpoint",
    ):
        assert getattr(graph, name, None) is getattr(training, name)


def test_teacher_example_owns_normal_cpu_training_tensors():
    with torch.inference_mode():
        hidden = [torch.randn(3, 2), torch.randn(3, 2)]
        scores = torch.rand(2, 1, 1, 3)
        token_ids = torch.tensor([[4, 5, 6]])
        prefix_ids = torch.tensor([[1, 2]])

    example = _symbol("TeacherExample")(
        dataset_name="fineweb_10k",
        dataset_index=7,
        token_ids=token_ids,
        hidden_by_layer=hidden,
        teacher_scores=scores,
        prefix_ids=prefix_ids,
        sequence_length=3,
    )

    assert example.dataset_name == "fineweb_10k"
    assert example.dataset_index == 7
    assert len(example.hidden_by_layer) == 2
    for tensor in (*example.hidden_by_layer, example.teacher_scores):
        assert tensor.device.type == "cpu"
        assert not torch.is_inference(tensor)
        assert not tensor.requires_grad
    torch.testing.assert_close(example.hidden_by_layer[0], hidden[0])
    assert example.token_ids.data_ptr() != token_ids.data_ptr()
    assert example.prefix_ids.data_ptr() != prefix_ids.data_ptr()


def test_teacher_example_rejects_inconsistent_context_lengths():
    with pytest.raises(ValueError, match="sequence length"):
        _symbol("TeacherExample")(
            dataset_name="fineweb_10k",
            dataset_index=0,
            token_ids=torch.arange(3).unsqueeze(0),
            hidden_by_layer=[torch.zeros(2, 4)],
            teacher_scores=torch.full((1, 1, 1, 3), 0.5),
            prefix_ids=torch.tensor([[1]]),
            sequence_length=3,
        )


def test_teacher_example_normalizes_real_hidden_cache_singleton_batch_dimension():
    with torch.inference_mode():
        cached_hidden = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]

    example = _symbol("TeacherExample")(
        dataset_name="fineweb_10k",
        dataset_index=0,
        token_ids=torch.arange(3).unsqueeze(0),
        hidden_by_layer=cached_hidden,
        teacher_scores=torch.full((2, 1, 1, 3), 0.5),
        prefix_ids=torch.tensor([[1]]),
        sequence_length=3,
    )

    assert [tensor.shape for tensor in example.hidden_by_layer] == [
        torch.Size([3, 4]),
        torch.Size([3, 4]),
    ]
    assert all(not torch.is_inference(tensor) for tensor in example.hidden_by_layer)


def test_scheduler_specs_validate_json_and_construct_standard_pytorch_scheduler():
    parse = _symbol("parse_scheduler_spec")
    build = _symbol("build_scheduler")
    spec = parse("StepLR", '{"step_size": 2, "gamma": 0.5}')
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    scheduler = build(optimizer, spec)
    optimizer.step()
    scheduler.step()

    assert spec.name == "StepLR"
    assert spec.kwargs == {"step_size": 2, "gamma": 0.5}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("NotAScheduler", "{}"),
        ("LambdaLR", "{}"),
        ("StepLR", "{}"),
        ("StepLR", "[]"),
        ("StepLR", "not-json"),
    ],
)
def test_scheduler_specs_reject_unknown_callable_required_or_invalid_constructors(
    name, kwargs
):
    with pytest.raises(ValueError):
        _symbol("parse_scheduler_spec")(name, kwargs)


def test_none_scheduler_rejects_orphan_kwargs():
    parse = _symbol("parse_scheduler_spec")
    assert parse(None, None) is None
    with pytest.raises(ValueError, match="without a scheduler"):
        parse(None, "{}")


def test_scheduler_mapping_still_rejects_non_json_callable_arguments():
    with pytest.raises(ValueError, match="JSON object"):
        _symbol("parse_scheduler_spec")(
            "LambdaLR", {"lr_lambda": lambda _step: 1.0}
        )


def test_joint_defaults_and_missing_values_copy_the_active_peer():
    resolve = _symbol("resolve_joint_settings")
    assert resolve(None, None, None, None) == (1e-4, 1e-4, None, None)

    cosine = _symbol("parse_scheduler_spec")("CosineAnnealingLR", '{"T_max": 4}')
    assert resolve(None, 3e-4, None, cosine) == (3e-4, 3e-4, cosine, cosine)


def test_joint_rejects_unequal_active_lr_or_scheduler_but_ignores_frozen_gate():
    parse = _symbol("parse_scheduler_spec")
    step = parse("StepLR", '{"step_size": 1}')
    cosine = parse("CosineAnnealingLR", '{"T_max": 2}')
    resolve = _symbol("resolve_joint_settings")

    with pytest.raises(ValueError, match="learning rates"):
        resolve(1e-4, 2e-4, step, step)
    with pytest.raises(ValueError, match="schedulers"):
        resolve(1e-4, 1e-4, step, cosine)

    assert resolve(1e-4, 2e-4, step, cosine, gate_frozen=True) == (
        1e-4,
        2e-4,
        step,
        cosine,
    )


@pytest.mark.parametrize("value", [0, -1e-4, float("inf"), float("nan"), True, "1e-4"])
def test_joint_rejects_invalid_learning_rates_before_model_loading(value):
    with pytest.raises(ValueError, match="learning rates"):
        _symbol("resolve_joint_settings")(value, None, None, None)


def test_b_init_auto_preserves_checkpoint_gate_baseline_and_unblocks_fresh_graph():
    resolve = _symbol("resolve_b_init")
    assert resolve("auto", has_gate_checkpoint=True) == "zero"
    assert resolve("auto", has_gate_checkpoint=False) == "random"
    assert resolve("zero", has_gate_checkpoint=False) == "zero"
    assert resolve("random", has_gate_checkpoint=True) == "random"
    with pytest.raises(ValueError, match="b-init"):
        resolve("identity", has_gate_checkpoint=False)


class TinyGate(nn.Module):
    def __init__(self, hidden_dim=3, heads=2, groups=2, gate_dim=2):
        super().__init__()
        self.nhead = heads
        self.ngroup = groups
        self.output_dim = gate_dim
        self.d = math.sqrt(gate_dim)
        self.q_proj = nn.Linear(hidden_dim, heads * groups * gate_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, heads * gate_dim, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.k_base = nn.Parameter(torch.randn(heads, 1, 2, gate_dim))
        self.b = nn.Parameter(torch.randn(heads, 1, groups))


class ChainBuilder(GraphBuilder):
    def __init__(self):
        super().__init__()
        self.edge_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, z):
        graph_count, token_count, _ = z.shape
        source = []
        target = []
        for graph in range(graph_count):
            offset = graph * token_count
            source.extend(offset + token for token in range(token_count - 1))
            target.extend(offset + token for token in range(1, token_count))
        edges = torch.tensor([source, target], device=z.device, dtype=torch.long)
        edge_index = EdgeIndex(
            edges,
            sparse_size=(graph_count * token_count, graph_count * token_count),
        )
        return GraphTopology(
            edge_index,
            self.edge_weight.expand(edge_index.size(1)),
        )


def _make_scorer_and_example(*, token_count=5):
    torch.manual_seed(23)
    scorer = GraphScorer(
        [TinyGate().double(), TinyGate().double()],
        SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2),
        graph_dim=2,
        graph_builder=ChainBuilder(),
        graph_microbatch_size=2,
    ).double()
    with torch.no_grad():
        scorer.b_proj.weight.normal_(std=0.15)
    hidden = torch.randn(2, token_count, 3, dtype=torch.float64)
    targets = torch.sigmoid(torch.randn(2, 1, 2, token_count, dtype=torch.float64))
    example = _symbol("TeacherExample")(
        dataset_name="fineweb_10k",
        dataset_index=1,
        token_ids=torch.arange(token_count).unsqueeze(0),
        hidden_by_layer=list(hidden),
        teacher_scores=targets,
        prefix_ids=torch.tensor([[9, 8]]),
        sequence_length=token_count,
    )
    return scorer, example, hidden, targets


def _named_gate_parameters(scorer):
    return {name: value for name, value in scorer.named_parameters() if name.startswith("gates.")}


def _named_graph_parameters(scorer):
    return {name: value for name, value in scorer.named_parameters() if not name.startswith("gates.")}


def _optimizer(parameters, *, lr=3e-3):
    return torch.optim.AdamW(list(parameters), lr=lr, weight_decay=0.02)


def _assert_named_tensors_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-10, atol=2e-12)


def test_staged_graph_phase_matches_naive_float64_gradients_and_adamw_step():
    naive, example, hidden, targets = _make_scorer_and_example()
    staged = copy.deepcopy(naive)
    for parameter in naive.gates.parameters():
        parameter.requires_grad_(False)
    naive_graph = _named_graph_parameters(naive)
    naive_optimizer = _optimizer(naive_graph.values())
    naive_loss = F.binary_cross_entropy(naive(hidden), targets)
    naive_loss.backward()
    naive_gradients = {name: parameter.grad.clone() for name, parameter in naive_graph.items()}
    naive_optimizer.step()

    staged_graph = _named_graph_parameters(staged)
    staged_optimizer = _optimizer(staged_graph.values())
    backward_calls = []
    hook = staged.gin.register_full_backward_hook(
        lambda *_args: backward_calls.append("gin")
    )
    result = _symbol("GraphTrainer")(
        staged,
        graph_optimizer=staged_optimizer,
        graph_microbatch_size=2,
        token_microbatch_size=2,
    ).train_graph_phase(example)
    hook.remove()

    assert result.loss == pytest.approx(naive_loss.item(), rel=1e-12, abs=1e-12)
    assert result.optimizer_steps == 1
    assert len(backward_calls) == 2
    _assert_named_tensors_close(
        {name: parameter.grad for name, parameter in staged_graph.items()},
        naive_gradients,
    )
    _assert_named_tensors_close(
        {name: parameter.detach() for name, parameter in staged_graph.items()},
        {name: parameter.detach() for name, parameter in naive_graph.items()},
    )
    assert all(parameter.grad is None for parameter in staged.gates.parameters())


def test_staged_joint_phase_matches_naive_float64_gradients_and_two_optimizer_steps():
    naive, example, hidden, targets = _make_scorer_and_example(token_count=4)
    staged = copy.deepcopy(naive)
    naive_gate = _named_gate_parameters(naive)
    naive_graph = _named_graph_parameters(naive)
    naive_gate_optimizer = _optimizer(naive_gate.values())
    naive_graph_optimizer = _optimizer(naive_graph.values())
    naive_loss = F.binary_cross_entropy(naive(hidden), targets)
    naive_loss.backward()
    naive_gradients = {
        name: parameter.grad.clone() for name, parameter in naive.named_parameters()
    }
    naive_gate_optimizer.step()
    naive_graph_optimizer.step()

    staged_gate = _named_gate_parameters(staged)
    staged_graph = _named_graph_parameters(staged)
    staged_gate_optimizer = _optimizer(staged_gate.values())
    staged_graph_optimizer = _optimizer(staged_graph.values())
    result = _symbol("GraphTrainer")(
        staged,
        gate_optimizer=staged_gate_optimizer,
        graph_optimizer=staged_graph_optimizer,
        graph_microbatch_size=2,
        token_microbatch_size=1,
    ).train_graph_phase(example, joint=True)

    assert result.loss == pytest.approx(naive_loss.item(), rel=1e-12, abs=1e-12)
    assert result.optimizer_steps == 2
    _assert_named_tensors_close(
        {name: parameter.grad for name, parameter in staged.named_parameters()},
        naive_gradients,
    )
    _assert_named_tensors_close(
        {name: parameter.detach() for name, parameter in staged.named_parameters()},
        {name: parameter.detach() for name, parameter in naive.named_parameters()},
    )


def test_staged_mean_bce_and_gradients_do_not_depend_on_graph_or_token_microbatch():
    base, example, _, _ = _make_scorer_and_example()
    small = copy.deepcopy(base)
    large = copy.deepcopy(base)
    small_graph = _named_graph_parameters(small)
    large_graph = _named_graph_parameters(large)

    small_result = _symbol("GraphTrainer")(
        small,
        graph_optimizer=_optimizer(small_graph.values(), lr=0.0),
        graph_microbatch_size=1,
        token_microbatch_size=1,
    ).train_graph_phase(example)
    large_result = _symbol("GraphTrainer")(
        large,
        graph_optimizer=_optimizer(large_graph.values(), lr=0.0),
        graph_microbatch_size=4,
        token_microbatch_size=5,
    ).train_graph_phase(example)

    assert small_result.loss == pytest.approx(large_result.loss, rel=1e-12, abs=1e-12)
    _assert_named_tensors_close(
        {name: parameter.grad for name, parameter in small_graph.items()},
        {name: parameter.grad for name, parameter in large_graph.items()},
    )


def test_gate_phase_caches_graph_once_then_updates_only_gate_per_shuffled_token_batch():
    scorer, example, _, _ = _make_scorer_and_example()
    zero_delta_scorer = copy.deepcopy(scorer)
    with torch.no_grad():
        zero_delta_scorer.b_proj.weight.zero_()
    original_graph = {
        name: parameter.detach().clone()
        for name, parameter in _named_graph_parameters(scorer).items()
    }
    original_gate = {
        name: parameter.detach().clone()
        for name, parameter in _named_gate_parameters(scorer).items()
    }
    gate_optimizer = _optimizer(scorer.gates.parameters())
    gin_calls = []
    hook = scorer.gin.register_forward_hook(lambda *_args: gin_calls.append("gin"))
    mixed_hidden = []
    adapter_hook = scorer._gate_adapter.register_forward_pre_hook(
        lambda _module, args: mixed_hidden.append(args[2].detach().to("cpu"))
    )
    expected_order = torch.randperm(5, generator=torch.Generator().manual_seed(71))
    torch.manual_seed(71)

    result = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_microbatch_size=2,
        token_microbatch_size=2,
    ).train_gate_phase(example)
    hook.remove()
    adapter_hook.remove()
    torch.manual_seed(71)
    _symbol("GraphTrainer")(
        zero_delta_scorer,
        gate_optimizer=_optimizer(zero_delta_scorer.gates.parameters()),
        graph_microbatch_size=2,
        token_microbatch_size=2,
    ).train_gate_phase(example)

    assert result.optimizer_steps == 3
    assert len(gin_calls) == 2
    torch.testing.assert_close(
        torch.cat([mixed_hidden[0], mixed_hidden[4], mixed_hidden[8]]),
        example.hidden_by_layer[0][expected_order],
    )
    assert gate_optimizer.state[next(iter(scorer.gates.parameters()))]["step"].item() == 3
    _assert_named_tensors_close(
        {name: parameter.detach() for name, parameter in _named_graph_parameters(scorer).items()},
        original_graph,
    )
    assert any(
        not torch.equal(parameter, original_gate[name])
        for name, parameter in _named_gate_parameters(scorer).items()
    )
    assert any(
        not torch.equal(parameter, _named_gate_parameters(zero_delta_scorer)[name])
        for name, parameter in _named_gate_parameters(scorer).items()
    )
    assert all(parameter.grad is None for parameter in _named_graph_parameters(scorer).values())


def test_real_weight_mixed_precision_runs_gate_and_graph_bce():
    torch.manual_seed(79)
    gate = Weight(
        index=0,
        input_dim=4,
        output_dim=2,
        nhead=1,
        ngroup=1,
        dtype=torch.bfloat16,
        sink=2,
    )
    scorer = GraphScorer(
        [gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=2,
        graph_microbatch_size=1,
    )
    with torch.no_grad():
        scorer.b_proj.weight.normal_(std=0.1)
    example = _symbol("TeacherExample")(
        dataset_name="fineweb_10k",
        dataset_index=0,
        token_ids=torch.arange(4).unsqueeze(0),
        hidden_by_layer=[torch.randn(4, 4)],
        teacher_scores=torch.sigmoid(torch.randn(1, 1, 1, 4)),
        prefix_ids=torch.tensor([[1, 2]]),
        sequence_length=4,
    )
    trainer = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=_optimizer(scorer.gates.parameters()),
        graph_optimizer=_optimizer(_named_graph_parameters(scorer).values()),
        token_microbatch_size=2,
        graph_microbatch_size=1,
    )

    gate_result = trainer.train_gate_phase(example)
    graph_result = trainer.train_graph_phase(example)

    assert gate.q_proj.weight.dtype == torch.bfloat16
    assert gate.q_norm.weight.dtype == torch.float32
    assert math.isfinite(gate_result.loss)
    assert math.isfinite(graph_result.loss)


@pytest.mark.parametrize("phase", ["gate", "graph"])
def test_training_phases_never_request_full_hidden(phase, monkeypatch):
    scorer, example, _, _ = _make_scorer_and_example(token_count=5)
    trainer = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=_optimizer(scorer.gates.parameters()),
        graph_optimizer=_optimizer(_named_graph_parameters(scorer).values()),
        token_microbatch_size=2,
        graph_microbatch_size=2,
    )
    load_hidden_chunk = trainer._hidden
    requested_positions = []

    def reject_full_hidden(example, layer_ids, positions=None):
        assert positions is not None, "full hidden was requested"
        requested_positions.append(positions.clone())
        return load_hidden_chunk(example, layer_ids, positions)

    monkeypatch.setattr(trainer, "_hidden", reject_full_hidden)

    result = getattr(trainer, f"train_{phase}_phase")(example)

    assert math.isfinite(result.loss)
    assert requested_positions
    assert max(positions.numel() for positions in requested_positions) <= 2


def test_adamw_setup_partitions_gate_and_graph_parameters_and_honors_gate_freeze():
    scorer, _, _, _ = _make_scorer_and_example()
    build = _symbol("build_adamw_optimizers")

    gate_optimizer, graph_optimizer = build(
        scorer, gate_lr=2e-4, graph_lr=3e-3, weight_decay=0.04
    )

    assert isinstance(gate_optimizer, torch.optim.AdamW)
    assert isinstance(graph_optimizer, torch.optim.AdamW)
    assert gate_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert graph_optimizer.param_groups[0]["lr"] == pytest.approx(3e-3)
    assert graph_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.04)
    gate_ids = {id(parameter) for group in gate_optimizer.param_groups for parameter in group["params"]}
    graph_ids = {
        id(parameter) for group in graph_optimizer.param_groups for parameter in group["params"]
    }
    assert gate_ids == {id(parameter) for parameter in scorer.gates.parameters()}
    assert graph_ids == {
        id(parameter) for parameter in _named_graph_parameters(scorer).values()
    }
    assert gate_ids.isdisjoint(graph_ids)

    frozen_gate, _ = build(scorer, gate_frozen=True)
    assert frozen_gate is None
    assert all(not parameter.requires_grad for parameter in scorer.gates.parameters())


def test_normal_scheduler_steps_after_optimizer_but_plateau_waits_for_validation():
    scorer, example, _, _ = _make_scorer_and_example(token_count=3)
    gate_optimizer = _optimizer(scorer.gates.parameters(), lr=0.08)
    graph_optimizer = _optimizer(_named_graph_parameters(scorer).values(), lr=0.06)
    gate_scheduler = torch.optim.lr_scheduler.StepLR(
        gate_optimizer, step_size=1, gamma=0.5
    )
    graph_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        graph_optimizer, factor=0.5, patience=0
    )
    trainer = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
        token_microbatch_size=2,
        graph_microbatch_size=2,
    )

    trainer.train_gate_phase(example)
    trainer.train_graph_phase(example)

    assert gate_optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert graph_optimizer.param_groups[0]["lr"] == pytest.approx(0.06)
    trainer.step_validation(0.4)
    assert graph_optimizer.param_groups[0]["lr"] == pytest.approx(0.06)
    trainer.step_validation(0.5)
    assert graph_optimizer.param_groups[0]["lr"] == pytest.approx(0.03)
    assert gate_optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_b_initialization_applies_resolved_zero_or_random_weights():
    scorer, _, _, _ = _make_scorer_and_example()
    initialize = _symbol("initialize_b_projection")

    assert initialize(scorer, "auto", has_gate_checkpoint=True) == "zero"
    assert torch.count_nonzero(scorer.b_proj.weight) == 0
    assert initialize(scorer, "auto", has_gate_checkpoint=False) == "random"
    assert torch.count_nonzero(scorer.b_proj.weight) > 0
    initialize(scorer, "zero", has_gate_checkpoint=False)
    assert torch.count_nonzero(scorer.b_proj.weight) == 0


@pytest.mark.parametrize(
    ("mode", "expected_gate_steps", "expected_graph_steps"),
    [("gate", 2, 0), ("graph", 0, 1), ("two_phase", 2, 1), ("joint", 1, 1)],
)
def test_context_modes_take_the_required_optimizer_steps(
    mode, expected_gate_steps, expected_graph_steps
):
    scorer, example, _, _ = _make_scorer_and_example(token_count=3)
    gate_optimizer = _optimizer(scorer.gates.parameters())
    graph_optimizer = _optimizer(_named_graph_parameters(scorer).values())
    trainer = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        token_microbatch_size=2,
        graph_microbatch_size=2,
    )

    result = trainer.train_context(example, mode=mode)

    gate_parameter = next(iter(scorer.gates.parameters()))
    graph_parameter = next(iter(_named_graph_parameters(scorer).values()))
    actual_gate_steps = gate_optimizer.state.get(gate_parameter, {}).get("step", 0)
    actual_graph_steps = graph_optimizer.state.get(graph_parameter, {}).get("step", 0)
    if isinstance(actual_gate_steps, torch.Tensor):
        actual_gate_steps = actual_gate_steps.item()
    if isinstance(actual_graph_steps, torch.Tensor):
        actual_graph_steps = actual_graph_steps.item()
    assert actual_gate_steps == expected_gate_steps
    assert actual_graph_steps == expected_graph_steps
    assert result["gate_steps"] == expected_gate_steps
    assert result["graph_steps"] == expected_graph_steps


@pytest.mark.parametrize("mode", ["two_phase", "joint"])
def test_context_modes_with_frozen_gate_take_only_one_graph_step(mode):
    scorer, example, _, _ = _make_scorer_and_example(token_count=3)
    original_gate = copy.deepcopy(scorer.gates.state_dict())
    gate_optimizer, graph_optimizer = _symbol("build_adamw_optimizers")(
        scorer, gate_frozen=True
    )
    result = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        token_microbatch_size=2,
        graph_microbatch_size=2,
    ).train_context(example, mode=mode)

    assert result["gate_steps"] == 0
    assert result["graph_steps"] == 1
    _assert_nested_equal(scorer.gates.state_dict(), original_gate)


def test_gate_only_mode_requires_an_active_gate_optimizer():
    scorer, example, _, _ = _make_scorer_and_example(token_count=3)
    trainer = _symbol("GraphTrainer")(
        scorer,
        graph_optimizer=_optimizer(_named_graph_parameters(scorer).values()),
    )

    with pytest.raises(ValueError, match="gate optimizer"):
        trainer.train_context(example, mode="gate")


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_best_and_last_checkpoint_round_trip_restores_complete_training_state_and_rng(
    tmp_path,
):
    scorer, example, _, _ = _make_scorer_and_example(token_count=3)
    gate_optimizer = _optimizer(scorer.gates.parameters())
    graph_optimizer = _optimizer(_named_graph_parameters(scorer).values())
    gate_scheduler = torch.optim.lr_scheduler.StepLR(
        gate_optimizer, step_size=1, gamma=0.8
    )
    graph_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        graph_optimizer, factor=0.5, patience=0
    )
    trainer = _symbol("GraphTrainer")(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
        token_microbatch_size=2,
        graph_microbatch_size=2,
    )
    trainer.train_context(example, mode="joint")
    trainer.step_validation(0.4)
    saved_model = copy.deepcopy(scorer.state_dict())
    saved_gate_optimizer = copy.deepcopy(gate_optimizer.state_dict())
    saved_graph_optimizer = copy.deepcopy(graph_optimizer.state_dict())
    saved_gate_scheduler = copy.deepcopy(gate_scheduler.state_dict())
    saved_graph_scheduler = copy.deepcopy(graph_scheduler.state_dict())
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    save = _symbol("save_checkpoint")
    common = dict(
        scorer=scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
        config={"mode": "joint", "graph_dim": 2},
        model_id="tiny/model",
        prefix_ids=torch.tensor([[11, 12, 13]]),
        prefill_chunk=4096,
        data_cursor={"epoch": 2, "dataset": "fineweb_10k", "index": 4},
        wandb_run_id="run-abc",
    )
    last_path = save(tmp_path, "last", **common)
    best_path = save(tmp_path, "best", **common)
    expected_random = (random.random(), np.random.rand(), torch.rand(3))

    assert last_path == tmp_path / "last.pt"
    assert best_path == tmp_path / "best.pt"
    payload_on_disk = torch.load(last_path, weights_only=False)
    assert {
        "graph",
        "gate",
        "graph_optimizer",
        "gate_optimizer",
        "graph_scheduler",
        "gate_scheduler",
        "config",
        "model_id",
        "prefix_ids",
        "prefill_chunk",
        "data_cursor",
        "rng",
        "wandb_run_id",
    } <= payload_on_disk.keys()

    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.add_(10)
    gate_optimizer.param_groups[0]["lr"] = 0.9
    graph_optimizer.param_groups[0]["lr"] = 0.8
    gate_scheduler.step()
    trainer.step_validation(0.8)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)

    payload = _symbol("load_checkpoint")(
        last_path,
        scorer=scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
    )

    _assert_nested_equal(scorer.state_dict(), saved_model)
    _assert_nested_equal(gate_optimizer.state_dict(), saved_gate_optimizer)
    _assert_nested_equal(graph_optimizer.state_dict(), saved_graph_optimizer)
    _assert_nested_equal(gate_scheduler.state_dict(), saved_gate_scheduler)
    _assert_nested_equal(graph_scheduler.state_dict(), saved_graph_scheduler)
    assert payload["config"] == {"mode": "joint", "graph_dim": 2}
    assert payload["model_id"] == "tiny/model"
    assert payload["prefill_chunk"] == 4096
    assert payload["data_cursor"] == {
        "epoch": 2,
        "dataset": "fineweb_10k",
        "index": 4,
    }
    assert payload["wandb_run_id"] == "run-abc"
    torch.testing.assert_close(payload["prefix_ids"], torch.tensor([[11, 12, 13]]))
    actual_random = (random.random(), np.random.rand(), torch.rand(3))
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)


def test_gate_checkpoint_accepts_fastkvzip_module_lists_and_training_payloads(tmp_path):
    source, _, _, _ = _make_scorer_and_example()
    target, _, _, _ = _make_scorer_and_example()
    with torch.no_grad():
        for parameter in source.gates.parameters():
            parameter.normal_()
        for parameter in target.gates.parameters():
            parameter.zero_()
    legacy_path = tmp_path / "legacy-gate.pt"
    torch.save({"module": [gate.state_dict() for gate in source.gates]}, legacy_path)

    load_gate = _symbol("load_gate_checkpoint")
    load_gate(target, legacy_path)
    _assert_nested_equal(target.gates.state_dict(), source.gates.state_dict())

    training_path = _symbol("save_checkpoint")(
        tmp_path,
        "last",
        scorer=source,
        config={},
        model_id="tiny/model",
        prefix_ids=torch.tensor([[1]]),
        prefill_chunk=2,
        data_cursor={},
        wandb_run_id=None,
    )
    with torch.no_grad():
        for parameter in target.gates.parameters():
            parameter.zero_()
    load_gate(target, training_path)
    _assert_nested_equal(target.gates.state_dict(), source.gates.state_dict())
