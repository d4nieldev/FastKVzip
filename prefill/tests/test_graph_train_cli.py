import argparse
import copy
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric import EdgeIndex

from graph import GraphBuilder, GraphScorer, GraphTopology, GraphTrainer, TeacherExample

try:
    import train_graph
except ImportError:
    train_graph = None


def _symbol(name):
    assert train_graph is not None, "train_graph.py is not implemented"
    value = getattr(train_graph, name, None)
    assert value is not None, f"{name} is not implemented"
    return value


class TinyGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 1
        self.ngroup = 1
        self.output_dim = 2
        self.sink = 2
        self.d = math.sqrt(2)
        self.q_proj = nn.Linear(3, 2, bias=True).double()
        self.k_proj = nn.Linear(3, 2, bias=False).double()
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.k_base = nn.Parameter(torch.randn(1, 1, 2, 2, dtype=torch.float64))
        self.b = nn.Parameter(torch.randn(1, 1, 1, dtype=torch.float64))


class ChainBuilder(GraphBuilder):
    def forward(self, z):
        _, token_count, _ = z.shape
        source = torch.arange(token_count - 1, device=z.device)
        target = source + 1
        edges = EdgeIndex(
            torch.stack([source, target]), sparse_size=(token_count, token_count)
        )
        return GraphTopology(edges)


def _scorer_and_example(token_count=5):
    torch.manual_seed(211)
    scorer = GraphScorer(
        [TinyGate()],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=2,
        graph_builder=ChainBuilder(),
        graph_microbatch_size=1,
    ).double()
    with torch.no_grad():
        scorer.b_proj.weight.normal_(std=0.1)
    example = TeacherExample(
        dataset_name="fineweb_10k",
        dataset_index=0,
        token_ids=torch.arange(token_count).unsqueeze(0),
        hidden_by_layer=[torch.randn(token_count, 3, dtype=torch.float64)],
        teacher_scores=torch.sigmoid(
            torch.randn(1, 1, 1, token_count, dtype=torch.float64)
        ),
        prefix_ids=torch.tensor([[90, 91]]),
        sequence_length=token_count,
    )
    return scorer, example


def test_dataset_keys_are_exact_and_ordered():
    assert _symbol("TRAIN_KEYS") == tuple(
        [("fineweb_10k", index) for index in range(29)]
        + [("fineweb_10k_cat", index) for index in range(5)]
    )


def test_parser_resolves_documented_defaults():
    options = _symbol("resolve_options")(_minimal_args())

    assert options.gate_dim == 16
    assert options.gate_sink == 16
    assert options.graph_dim == 32
    assert options.gin_depth == 1
    assert options.num_neighbors == 16
    assert options.graph_microbatch_size == "auto"
    assert options.token_microbatch_size == 1000
    assert options.knn_index == "ivf_flat"
    assert options.ivf_nlist == 256
    assert options.ivf_nprobe == 16
    assert options.ivfpq_m == 8
    assert options.ivfpq_bits == 8
    assert options.mode == "two_phase"
    assert options.gate_lr == pytest.approx(1e-4)
    assert options.graph_lr == pytest.approx(1e-3)
    assert options.prefill_chunk == 16000
    assert options.b_init == "auto"
    assert options.epochs == 1
    assert options.max_contexts is None
    assert options.wandb_mode == "online"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_max_contexts_must_be_positive_before_wandb_or_model_loading(value):
    events = []

    class UntouchedWandb:
        @staticmethod
        def init(**_kwargs):
            events.append("wandb")

    with pytest.raises(ValueError, match="max-contexts must be positive"):
        _symbol("run_training")(
            _minimal_args("--max-contexts", value),
            model_factory=lambda *_args, **_kwargs: events.append("model"),
            wandb_module=UntouchedWandb,
        )

    assert events == []


@pytest.mark.parametrize("mode", ["gate", "graph"])
def test_public_cli_rejects_internal_single_phase_modes(mode):
    with pytest.raises(SystemExit):
        _minimal_args("--training-mode", mode)


def test_resume_rejects_saved_internal_single_phase_mode():
    args = _minimal_args("--resume", "checkpoint.pt")
    payload = {
        "model_id": "tiny/model",
        "prefill_chunk": 16000,
        "config": {"training_mode": "gate"},
    }

    with pytest.raises(ValueError, match="two_phase or joint"):
        _symbol("resolve_options")(args, resume_payload=payload)
    assert _symbol("VALIDATION_KEYS") == tuple(
        [("fineweb_10k", index) for index in range(29, 32)]
        + [("fineweb_10k_cat", 5)]
    )


def test_teacher_is_constructed_with_retain_cache_and_no_builtin_gate():
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    teacher = _symbol("build_teacher")("tiny/model", model_factory=factory)

    assert teacher is not None
    assert calls == [(("tiny/model",), {"kv_type": "retain", "gate_path_or_name": ""})]


def test_teacher_example_slices_only_context_and_clones_inference_tensors():
    with torch.inference_mode():
        hidden = [
            torch.arange(30, dtype=torch.float32).view(1, 5, 6),
            torch.arange(30, 60, dtype=torch.float32).view(1, 5, 6),
        ]
        score = [
            torch.tensor([[[0.1, 0.2, 0.3]]]),
            torch.tensor([[[0.4, 0.5, 0.6]]]),
        ]
        kv = SimpleNamespace(
            start_idx=2,
            end_idx=5,
            hidden_cache=hidden,
            score=score,
            prefill_ids=torch.tensor([[7, 8, 10, 11, 12]]),
        )

    example = _symbol("teacher_example_from_kv")(kv, "fineweb_10k", 4)

    assert example.sequence_length == 3
    torch.testing.assert_close(example.token_ids, torch.tensor([[10, 11, 12]]))
    torch.testing.assert_close(example.prefix_ids, torch.tensor([[7, 8]]))
    assert example.teacher_scores.shape == (2, 1, 1, 3)
    assert [tensor.shape for tensor in example.hidden_by_layer] == [(3, 6), (3, 6)]
    assert all(tensor.device.type == "cpu" for tensor in example.hidden_by_layer)
    assert all(not torch.is_inference(tensor) for tensor in example.hidden_by_layer)
    with torch.inference_mode():
        hidden[0][0, 2, 0] = -100
    assert example.hidden_by_layer[0][0, 0].item() != -100


@pytest.mark.parametrize("stacked", [False, True])
def test_teacher_example_supports_singleton_layer_head_token_and_dimension(stacked):
    score = torch.tensor([[[[0.25]]]])
    kv = SimpleNamespace(
        start_idx=1,
        end_idx=2,
        hidden_cache=[torch.tensor([[[7.0], [8.0]]])],
        score=score if stacked else [score[0]],
        prefill_ids=torch.tensor([[5, 6]]),
    )

    example = _symbol("teacher_example_from_kv")(kv, "fineweb_10k", 0)

    assert example.hidden_by_layer[0].shape == (1, 1)
    assert example.teacher_scores.shape == (1, 1, 1, 1)
    assert example.token_ids.shape == (1, 1)
    torch.testing.assert_close(example.hidden_by_layer[0], torch.tensor([[8.0]]))


def test_cursor_points_to_next_context_and_preserves_partial_validation():
    initial = _symbol("initial_cursor")()
    assert _symbol("next_context_key")(initial) == ("fineweb_10k", 0)

    last_train = {
        **initial,
        "phase": "train",
        "offset": len(_symbol("TRAIN_KEYS")) - 1,
        "wandb_step": 33,
    }
    after_train, completed = _symbol("advance_cursor")(last_train)
    assert completed is None
    assert after_train["phase"] == "validation"
    assert after_train["offset"] == 0
    assert after_train["wandb_step"] == 34
    assert _symbol("next_context_key")(after_train) == ("fineweb_10k", 29)

    partial = {
        **after_train,
        "offset": 2,
        "validation_sum": 0.6,
        "validation_count": 2,
        "wandb_step": 36,
    }
    assert _symbol("next_context_key")(partial) == ("fineweb_10k", 31)
    resumed, completed = _symbol("advance_cursor")(partial, validation_loss=0.2)
    assert completed is None
    assert resumed["offset"] == 3
    assert resumed["validation_sum"] == pytest.approx(0.8)
    assert resumed["validation_count"] == 3
    final, completed = _symbol("advance_cursor")(resumed, validation_loss=0.4)
    assert completed == pytest.approx(0.3)
    assert final["epoch"] == 1
    assert final["phase"] == "train"
    assert final["offset"] == 0
    assert final["validation_sum"] == 0
    assert final["validation_count"] == 0
    assert final["best_validation_bce"] == pytest.approx(0.3)
    assert final["wandb_step"] == 38


class FakeRun:
    def __init__(self):
        self.id = "run-1"
        self.logged = []
        self.finished = False

    def log(self, metrics, step=None):
        self.logged.append((copy.deepcopy(metrics), step))

    def finish(self):
        self.finished = True


def test_one_wandb_log_per_context_even_with_many_gate_updates():
    scorer, example = _scorer_and_example(token_count=5)
    gate_optimizer = torch.optim.AdamW(scorer.gates.parameters(), lr=1e-3)
    graph_parameters = [
        parameter
        for name, parameter in scorer.named_parameters()
        if not name.startswith("gates.")
    ]
    graph_optimizer = torch.optim.AdamW(graph_parameters, lr=2e-3)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        token_microbatch_size=2,
        graph_microbatch_size=1,
    )
    run = FakeRun()

    result, metrics = _symbol("run_and_log_context")(
        trainer,
        example,
        mode="two_phase",
        validation=False,
        run=run,
        step=7,
    )

    assert result["gate_steps"] == 3
    assert len(run.logged) == 1
    assert run.logged[0][1] == 7
    assert run.logged[0][0] == metrics
    assert {
        "gate/bce",
        "graph/bce",
        "delta_energy_share",
        "gate/forward_seconds",
        "gate/backward_seconds",
        "graph/forward_seconds",
        "graph/backward_seconds",
        "gpu/peak_allocated_bytes",
        "gpu/peak_reserved_bytes",
        "gate/learning_rate",
        "graph/learning_rate",
    } == metrics.keys()
    assert all(
        parameter.grad is None
        for optimizer in (gate_optimizer, graph_optimizer)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _minimal_args(*extra):
    parser = _symbol("build_parser")()
    return parser.parse_args(["--model", "tiny/model", *extra])


def test_online_wandb_failure_occurs_before_model_loading():
    calls = []

    class FailingWandb:
        @staticmethod
        def login():
            raise RuntimeError("bad credentials")

    args = _minimal_args("--wandb-mode", "online")

    with pytest.raises(RuntimeError, match="bad credentials"):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: calls.append("model"),
            wandb_module=FailingWandb,
        )

    assert calls == []


def test_online_wandb_init_failure_also_occurs_before_model_loading():
    calls = []

    class FailingWandb:
        @staticmethod
        def login():
            return True

        @staticmethod
        def init(**_kwargs):
            raise RuntimeError("init failed")

    with pytest.raises(RuntimeError, match="init failed"):
        _symbol("run_training")(
            _minimal_args("--wandb-mode", "online"),
            model_factory=lambda *_args, **_kwargs: calls.append("model"),
            wandb_module=FailingWandb,
        )

    assert calls == []


def test_model_dependent_microbatch_validation_precedes_dataset_or_teacher_generation():
    events = []

    class DisabledWandb:
        @staticmethod
        def init(**_kwargs):
            events.append("wandb")
            return FakeRun()

    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=3,
    )
    teacher = SimpleNamespace(config=config)
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--graph-microbatch-size",
        "2",
    )

    with pytest.raises(ValueError, match="graph microbatch"):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: events.append("model") or teacher,
            dataset_loader=lambda *_args, **_kwargs: events.append("dataset"),
            wandb_module=DisabledWandb,
        )

    assert events == ["wandb", "model"]


def test_pure_scheduler_and_freeze_validation_precedes_model_loading():
    calls = []
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--gate-lr-scheduler",
        "NotAScheduler",
    )
    with pytest.raises(ValueError, match="unknown PyTorch scheduler"):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: calls.append("model"),
        )
    assert calls == []

    args = _minimal_args("--wandb-mode", "disabled", "--freeze-gate")
    with pytest.raises(ValueError, match="freeze-gate"):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: calls.append("model"),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [("--gate-dim", "3", "gate-dim"), ("--gate-sink", "4", "gate-sink")],
)
def test_local_gate_checkpoint_dimension_conflicts_fail_before_model_loading(
    tmp_path, flag, value, message
):
    checkpoint = tmp_path / "gate.pt"
    torch.save(
        {
            "config": {"gate_dim": 2, "gate_sink": 2},
            "module": [],
        },
        checkpoint,
    )
    calls = []
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--gate-checkpoint",
        str(checkpoint),
        flag,
        value,
    )

    with pytest.raises(ValueError, match=message):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: calls.append("model"),
        )

    assert calls == []


def test_explicit_auto_pq_and_graph_microbatch_are_normalized():
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--graph-dim",
        "12",
        "--ivfpq-m",
        "auto",
        "--graph-microbatch-size",
        "auto",
    )

    options = _symbol("resolve_options")(args)

    assert options.ivfpq_m == 6
    assert options.graph_microbatch_size == "auto"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_ivfpq_m_fails_before_wandb_or_model(value):
    events = []

    class RecordingWandb:
        @staticmethod
        def init(**_kwargs):
            events.append("wandb")
            return FakeRun()

    with pytest.raises(ValueError, match="ivfpq-m"):
        _symbol("run_training")(
            _minimal_args(
                "--wandb-mode", "disabled", "--ivfpq-m", value
            ),
            model_factory=lambda *_args, **_kwargs: events.append("model"),
            wandb_module=RecordingWandb,
        )

    assert events == []


@pytest.mark.parametrize("resolved", ["zero", "random"])
def test_explicit_auto_b_init_on_resume_uses_checkpoint_resolution(resolved):
    args = _minimal_args(
        "--resume", "checkpoint.pt", "--b-init", "auto"
    )
    payload = {
        "model_id": "tiny/model",
        "prefill_chunk": 16000,
        "config": {"b_init": resolved},
    }

    options = _symbol("resolve_options")(args, resume_payload=payload)

    assert options.b_init == resolved


def test_cli_lr_and_scheduler_resolution_for_two_phase_and_joint_modes():
    two_phase = _symbol("resolve_options")(_minimal_args())
    assert (two_phase.gate_lr, two_phase.graph_lr) == pytest.approx((1e-4, 1e-3))

    copied = _symbol("resolve_options")(
        _minimal_args(
            "--training-mode",
            "joint",
            "--gate-lr",
            "0.0002",
            "--gate-lr-scheduler",
            "StepLR",
            "--gate-lr-scheduler-kwargs",
            '{"step_size": 2}',
        )
    )
    assert (copied.gate_lr, copied.graph_lr) == pytest.approx((2e-4, 2e-4))
    assert copied.gate_scheduler == copied.graph_scheduler
    assert copied.gate_scheduler.name == "StepLR"

    with pytest.raises(ValueError, match="learning rates"):
        _symbol("resolve_options")(
            _minimal_args(
                "--training-mode",
                "joint",
                "--gate-lr",
                "0.0001",
                "--graph-lr",
                "0.0002",
            )
        )
    with pytest.raises(ValueError, match="schedulers"):
        _symbol("resolve_options")(
            _minimal_args(
                "--training-mode",
                "joint",
                "--gate-lr-scheduler",
                "StepLR",
                "--gate-lr-scheduler-kwargs",
                '{"step_size": 2}',
                "--graph-lr-scheduler",
                "CosineAnnealingLR",
                "--graph-lr-scheduler-kwargs",
                '{"T_max": 2}',
            )
        )
    with pytest.raises(ValueError, match="schedulers"):
        _symbol("resolve_options")(
            _minimal_args(
                "--training-mode",
                "joint",
                "--gate-lr-scheduler",
                "none",
                "--graph-lr-scheduler",
                "StepLR",
                "--graph-lr-scheduler-kwargs",
                '{"step_size": 2}',
            )
        )


def test_normalized_checkpoint_config_is_plain_and_reconstructs_singleton_model():
    scorer, _ = _scorer_and_example(token_count=2)
    options = SimpleNamespace(
        mode="two_phase",
        graph_dim=2,
        gin_depth=1,
        graph_microbatch_size=1,
        num_neighbors=3,
        knn_index="ivf_flat",
        ivf_nlist=8,
        ivf_nprobe=2,
        ivfpq_m=1,
        ivfpq_bits=4,
        token_microbatch_size=2,
        gate_lr=1e-4,
        graph_lr=1e-3,
        gate_scheduler=None,
        graph_scheduler=None,
        b_init="zero",
        freeze_gate=False,
    )

    config = _symbol("normalized_checkpoint_config")(
        model_id="tiny/model",
        scorer=scorer,
        options=options,
        query_groups=1,
    )

    assert config == {
        "format_version": 1,
        "model_id": "tiny/model",
        "gate_dim": 2,
        "gate_sink": 2,
        "hidden_dim": 3,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 2,
        "gin_depth": 1,
        "graph_microbatch_size": 1,
        "num_neighbors": 3,
        "knn_index": "ivf_flat",
        "ivf_nlist": 8,
        "ivf_nprobe": 2,
        "ivfpq_m": 1,
        "ivfpq_bits": 4,
        "training_mode": "two_phase",
        "token_microbatch_size": 2,
        "gate_lr": 0.0001,
        "graph_lr": 0.001,
        "gate_lr_scheduler": None,
        "graph_lr_scheduler": None,
        "b_init": "zero",
        "freeze_gate": False,
    }
    assert all(
        value is None or isinstance(value, (str, int, float, bool, dict))
        for value in config.values()
    )


def _fake_teacher():
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=1,
    )
    return SimpleNamespace(
        config=config,
        dtype=torch.float32,
        device=torch.device("cpu"),
        model=SimpleNamespace(name_or_path="tiny/model"),
        tokenizer=object(),
        sys_prompt_ids=torch.tensor([[99]]),
    )


def test_local_checkpoint_auto_b_and_freeze_create_zero_b_without_gate_optimizer(
    tmp_path,
):
    from attention.gate import Weight

    gate = Weight(0, 1, 2, 1, 1, torch.float32, sink=2)
    checkpoint = tmp_path / "gate.pt"
    torch.save({"module": [gate.state_dict()]}, checkpoint)
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--gate-checkpoint",
        str(checkpoint),
        "--freeze-gate",
        "--graph-dim",
        "2",
        "--graph-microbatch-size",
        "auto",
    )
    payload = torch.load(checkpoint, weights_only=False)
    options = _symbol("resolve_options")(args, gate_payload=payload)

    options, scorer, trainer, config = _symbol("_make_components")(
        _fake_teacher(), options, None
    )

    assert options.b_init == "zero"
    assert config["b_init"] == "zero"
    assert config["graph_microbatch_size"] == 1
    assert torch.count_nonzero(scorer.b_proj.weight) == 0
    assert trainer.gate_optimizer is None
    assert trainer.graph_optimizer is not None


def test_gate_only_logging_omits_inactive_graph_metrics_and_optimizer():
    scorer, example = _scorer_and_example(token_count=3)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.AdamW(scorer.gates.parameters(), lr=1e-3),
        graph_optimizer=None,
        token_microbatch_size=2,
        graph_microbatch_size=1,
    )
    run = FakeRun()

    _, metrics = _symbol("run_and_log_context")(
        trainer,
        example,
        mode="gate",
        validation=False,
        run=run,
        step=0,
    )

    assert len(run.logged) == 1
    assert {
        "gate/bce",
        "delta_energy_share",
        "gate/forward_seconds",
        "gate/backward_seconds",
        "gpu/peak_allocated_bytes",
        "gpu/peak_reserved_bytes",
        "gate/learning_rate",
    } == metrics.keys()


def test_cuda_phase_timing_queues_events_and_synchronizes_once(monkeypatch):
    calls = []

    class Event:
        def record(self):
            calls.append("record")

        def elapsed_time(self, _other):
            return 250.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: Event())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("sync"))
    timing = _symbol("PhaseTiming")(torch.device("cuda"))

    with timing.region("graph", "forward"):
        pass
    result = timing.resolve()

    assert result == {"graph_forward_seconds": 0.25}
    assert calls.count("sync") == 1


def test_context_result_is_materialized_only_after_timing_resolution(monkeypatch):
    scorer, example = _scorer_and_example(token_count=2)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.AdamW(scorer.gates.parameters(), lr=1e-3),
        graph_optimizer=torch.optim.AdamW(
            [
                parameter
                for name, parameter in scorer.named_parameters()
                if not name.startswith("gates.")
            ],
            lr=1e-3,
        ),
        token_microbatch_size=1,
        graph_microbatch_size=1,
    )
    events = []

    class Timing:
        def __init__(self, _device):
            pass

        def region(self, _phase, _operation):
            return nullcontext()

        def resolve(self):
            events.append("resolve")
            return {}

    real_materialize = _symbol("_materialize_context_result")

    def materialize(result):
        events.append("materialize")
        return real_materialize(result)

    monkeypatch.setattr(train_graph, "PhaseTiming", Timing)
    monkeypatch.setattr(train_graph, "_materialize_context_result", materialize)

    result, _ = _symbol("run_and_log_context")(
        trainer,
        example,
        mode="joint",
        validation=False,
        run=FakeRun(),
        step=0,
    )

    assert events == ["resolve", "materialize"]
    assert isinstance(result["graph_loss"], float)


class _RecordingWandb:
    def __init__(self):
        self.init_calls = []
        self.runs = []
        self.watch_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = FakeRun()
        run.id = kwargs.get("id", "first-run")
        self.runs.append(run)
        return run

    def watch(self, *_args, **_kwargs):
        self.watch_calls += 1
        raise AssertionError("wandb.watch must not be called")


class _TinyWrapper:
    calls = []
    events = []
    fail_key = None

    def __init__(self, name, _dataset, teacher):
        self.name = name
        self.teacher = teacher

    def prefill_context(self, index, **kwargs):
        key = (self.name, index)
        self.events.append("prefill")
        self.calls.append((key, kwargs))
        if key == self.fail_key:
            raise RuntimeError("simulated interruption")
        token = 10 + index + (100 if "cat" in self.name else 0)
        return SimpleNamespace(
            start_idx=1,
            end_idx=2,
            hidden_cache=[torch.tensor([[[0.0], [float(token) / 100]]])],
            score=[torch.tensor([[[0.25 + (index % 2) * 0.1]]])],
            prefill_ids=torch.tensor([[99, token]]),
        )


def _tiny_limited_args(output_dir, *extra):
    return _minimal_args(
        "--wandb-mode",
        "disabled",
        "--output-dir",
        str(output_dir),
        "--graph-dim",
        "1",
        "--gate-dim",
        "1",
        "--gate-sink",
        "1",
        "--num-neighbors",
        "1",
        "--token-microbatch-size",
        "1",
        "--max-contexts",
        "1",
        *extra,
    )


def test_max_contexts_stops_after_one_saved_context(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_graph,
        "TRAIN_KEYS",
        (("fineweb_10k", 0), ("fineweb_10k", 1)),
    )
    monkeypatch.setattr(train_graph, "VALIDATION_KEYS", (("fineweb_10k", 2),))
    _TinyWrapper.calls = []
    _TinyWrapper.events = []
    _TinyWrapper.fail_key = None
    wandb_module = _RecordingWandb()
    output_dir = tmp_path / "limited"

    last_path = _symbol("run_training")(
        _tiny_limited_args(output_dir),
        model_factory=lambda *_args, **_kwargs: _fake_teacher(),
        dataset_loader=lambda *_args, **_kwargs: object(),
        wrapper_factory=_TinyWrapper,
        wandb_module=wandb_module,
    )

    assert last_path == output_dir / "last.pt"
    assert [key for key, _kwargs in _TinyWrapper.calls] == [("fineweb_10k", 0)]
    assert len(wandb_module.runs[0].logged) == 1
    assert wandb_module.runs[0].finished
    checkpoint = torch.load(last_path, weights_only=False)
    assert checkpoint["data_cursor"]["phase"] == "train"
    assert checkpoint["data_cursor"]["offset"] == 1
    assert checkpoint["data_cursor"]["wandb_step"] == 1
    assert "max_contexts" not in checkpoint["config"]
    assert not (output_dir / "best.pt").exists()


def test_max_contexts_is_invocation_local_when_resuming(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_graph,
        "TRAIN_KEYS",
        (("fineweb_10k", 0), ("fineweb_10k", 1), ("fineweb_10k", 2)),
    )
    monkeypatch.setattr(train_graph, "VALIDATION_KEYS", (("fineweb_10k", 3),))
    _TinyWrapper.calls = []
    _TinyWrapper.events = []
    _TinyWrapper.fail_key = None
    wandb_module = _RecordingWandb()
    output_dir = tmp_path / "resume-limited"
    _symbol("run_training")(
        _tiny_limited_args(output_dir),
        model_factory=lambda *_args, **_kwargs: _fake_teacher(),
        dataset_loader=lambda *_args, **_kwargs: object(),
        wrapper_factory=_TinyWrapper,
        wandb_module=wandb_module,
    )
    _TinyWrapper.calls = []

    last_path = _symbol("run_training")(
        _minimal_args(
            "--wandb-mode",
            "disabled",
            "--output-dir",
            str(output_dir),
            "--resume",
            str(output_dir / "last.pt"),
            "--max-contexts",
            "1",
        ),
        model_factory=lambda *_args, **_kwargs: _fake_teacher(),
        dataset_loader=lambda *_args, **_kwargs: object(),
        wrapper_factory=_TinyWrapper,
        wandb_module=wandb_module,
    )

    assert last_path == output_dir / "last.pt"
    assert [key for key, _kwargs in _TinyWrapper.calls] == [("fineweb_10k", 1)]
    assert len(wandb_module.runs[1].logged) == 1
    assert wandb_module.runs[1].finished
    checkpoint = torch.load(last_path, weights_only=False)
    assert checkpoint["data_cursor"]["phase"] == "train"
    assert checkpoint["data_cursor"]["offset"] == 2
    assert checkpoint["data_cursor"]["wandb_step"] == 2


def test_run_training_resume_starts_at_next_partial_validation_context(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(train_graph, "TRAIN_KEYS", (("fineweb_10k", 0),))
    monkeypatch.setattr(
        train_graph,
        "VALIDATION_KEYS",
        (("fineweb_10k", 1), ("fineweb_10k_cat", 0)),
    )
    real_reset = train_graph._reset_peak_memory_stats
    monkeypatch.setattr(
        train_graph,
        "_reset_peak_memory_stats",
        lambda device: _TinyWrapper.events.append("reset") or real_reset(device),
    )
    wandb_module = _RecordingWandb()
    _TinyWrapper.calls = []
    _TinyWrapper.events = []
    _TinyWrapper.fail_key = ("fineweb_10k_cat", 0)
    output_dir = tmp_path / "run"
    args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--output-dir",
        str(output_dir),
        "--graph-dim",
        "1",
        "--gate-dim",
        "1",
        "--gate-sink",
        "1",
        "--num-neighbors",
        "1",
        "--token-microbatch-size",
        "1",
        "--graph-lr-scheduler",
        "ReduceLROnPlateau",
        "--graph-lr-scheduler-kwargs",
        '{"factor": 0.5, "patience": 0}',
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _symbol("run_training")(
            args,
            model_factory=lambda *_args, **_kwargs: _fake_teacher(),
            dataset_loader=lambda *_args, **_kwargs: object(),
            wrapper_factory=_TinyWrapper,
            wandb_module=wandb_module,
        )

    partial = torch.load(output_dir / "last.pt", weights_only=False)
    assert partial["data_cursor"]["phase"] == "validation"
    assert partial["data_cursor"]["offset"] == 1
    assert partial["data_cursor"]["validation_count"] == 1
    assert partial["data_cursor"]["wandb_step"] == 2
    assert len(wandb_module.runs[0].logged) == 2
    assert _TinyWrapper.events == [
        "reset",
        "prefill",
        "reset",
        "prefill",
        "reset",
        "prefill",
    ]

    _TinyWrapper.calls = []
    _TinyWrapper.fail_key = None
    resume_args = _minimal_args(
        "--wandb-mode",
        "disabled",
        "--output-dir",
        str(output_dir),
        "--resume",
        str(output_dir / "last.pt"),
    )
    real_load = torch.load
    loaded_paths = []

    def counting_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    last_path = _symbol("run_training")(
        resume_args,
        model_factory=lambda *_args, **_kwargs: _fake_teacher(),
        dataset_loader=lambda *_args, **_kwargs: object(),
        wrapper_factory=_TinyWrapper,
        wandb_module=wandb_module,
    )

    assert last_path == output_dir / "last.pt"
    assert loaded_paths.count(output_dir / "last.pt") == 1
    assert [key for key, _kwargs in _TinyWrapper.calls] == [
        ("fineweb_10k_cat", 0)
    ]
    assert wandb_module.init_calls[1]["id"] == "first-run"
    assert wandb_module.init_calls[1]["resume"] == "allow"
    assert len(wandb_module.runs[1].logged) == 1
    assert wandb_module.runs[1].logged[0][1] == 2
    assert wandb_module.watch_calls == 0
    completed = real_load(output_dir / "last.pt", weights_only=False)
    best = real_load(output_dir / "best.pt", weights_only=False)
    assert completed["data_cursor"]["epoch"] == 1
    assert completed["data_cursor"]["phase"] == "train"
    assert completed["data_cursor"]["wandb_step"] == 3
    assert best["data_cursor"] == completed["data_cursor"]
    assert math.isfinite(completed["graph_scheduler"]["best"])
    assert completed["config"]["graph_microbatch_size"] == 1
    assert completed["config"]["ivfpq_m"] == 1
    assert completed["config"]["b_init"] == "random"
