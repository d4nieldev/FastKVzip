import weakref
from types import SimpleNamespace

import pytest
import torch

import train_graph
from graph import TeacherExample


def _args(*extra):
    return train_graph.build_parser().parse_args(["--model", "unit", *extra])


def _example(index=0):
    return TeacherExample.from_owned_cpu(
        dataset_name="fineweb_10k",
        dataset_index=index,
        token_ids=torch.arange(3).view(1, -1),
        hidden_by_layer=[torch.randn(3, 2)],
        teacher_scores=torch.rand(1, 1, 1, 3),
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        sequence_length=3,
    )


def _training_data(train_keys, validation_keys=()):
    datasets = {}
    for name, index in (*train_keys, *validation_keys):
        datasets.setdefault(name, {})[index] = {}
    return datasets, tuple(train_keys), tuple(validation_keys)


def test_cli_defaults_are_joint_implicit_mixer_defaults():
    options = train_graph.resolve_options(_args())
    assert options.mode == "joint"
    assert options.graph_dim == 32
    assert options.gram_normalization == "token-count"
    assert options.leaky_relu_slope == pytest.approx(0.01)
    assert options.alpha_init == pytest.approx(0.1)
    assert options.gate_lr == pytest.approx(1e-4)
    assert options.mixer_lr == pytest.approx(1e-3)
    assert options.adamw_eps == pytest.approx(1e-8)
    assert not options.amsgrad
    assert options.graph_microbatch_size == "auto"
    assert options.subgraph_size is None
    assert options.teacher_cache_dir is None
    assert options.gate_scheduler is None
    assert options.mixer_scheduler is None
    assert options.save_strategy == "epochs"
    assert options.save_every == 1
    assert options.save_best
    assert options.eval_strategy == "epochs"
    assert options.eval_every == 1


def test_train_context_count_defaults_to_29_and_is_customizable():
    assert train_graph.resolve_options(_args()).train_context_count == 29
    custom = train_graph.resolve_options(_args("--train-context-count", "100"))
    assert custom.train_context_count == 100


def test_train_context_count_must_be_positive():
    with pytest.raises(ValueError, match="train-context-count.*positive"):
        train_graph.resolve_options(_args("--train-context-count", "0"))


@pytest.mark.parametrize(
    "option",
    (
        "--train-data-start-idx",
        "--train-data-end-idx",
        "--train-data-cat-start-idx",
        "--train-data-cat-end-idx",
        "--val-data-start-idx",
        "--val-data-end-idx",
        "--val-data-cat-start-idx",
        "--val-data-cat-end-idx",
    ),
)
def test_old_data_range_options_are_removed(option):
    with pytest.raises(SystemExit):
        train_graph.build_parser().parse_args(["--model", "unit", option, "0"])


def test_joint_allows_independent_learning_rates_and_schedulers():
    options = train_graph.resolve_options(
        _args(
            "--gate-lr", "0.0002",
            "--mixer-lr", "0.003",
            "--gate-lr-scheduler", "StepLR",
            "--gate-lr-scheduler-kwargs", '{"step_size": 1}',
            "--mixer-lr-scheduler", "ExponentialLR",
            "--mixer-lr-scheduler-kwargs", '{"gamma": 0.9}',
        )
    )
    assert options.gate_lr == pytest.approx(2e-4)
    assert options.mixer_lr == pytest.approx(3e-3)
    assert options.gate_scheduler.name == "StepLR"
    assert options.mixer_scheduler.name == "ExponentialLR"


def test_adamw_options_are_shared_and_validated():
    options = train_graph.resolve_options(_args("--adamw-eps", "1e-4", "--amsgrad"))
    assert options.adamw_eps == pytest.approx(1e-4)
    assert options.amsgrad
    with pytest.raises(ValueError, match="AdamW epsilon.*positive"):
        train_graph.resolve_options(_args("--adamw-eps", "0"))
    legacy = train_graph.resolve_options(
        _args(), {"model_id": "unit", "config": {}, "prefill_chunk": 16000}
    )
    assert legacy.adamw_eps == pytest.approx(1e-8)
    assert not legacy.amsgrad


def test_two_phase_rejects_context_scheduled_trainable_gate():
    with pytest.raises(ValueError, match="joint training"):
        train_graph.resolve_options(
            _args(
                "--mode", "two_phase",
                "--gate-lr-scheduler", "LinearWarmupCosineLR",
                "--gate-lr-scheduler-kwargs", '{"warmup_fraction": 0.5}',
            )
        )


def test_subgraph_cli_requires_joint_mode_and_divisible_token_batches():
    options = train_graph.resolve_options(
        _args("--subgraph-size", "2000", "--token-microbatch-size", "16000")
    )
    assert options.subgraph_size == 2000
    for extra in (
        ("--subgraph-size", "0"),
        ("--subgraph-size", "2000", "--token-microbatch-size", "3000"),
        ("--subgraph-size", "2000", "--token-microbatch-size", "16000", "--mode", "two_phase"),
    ):
        with pytest.raises(ValueError, match="subgraph"):
            train_graph.resolve_options(_args(*extra))


def test_subgraph_size_is_a_legacy_compatible_resume_invariant():
    saved = {
        "model_id": "unit",
        "prefill_chunk": 16000,
        "config": {
            "subgraph_size": 2000,
            "token_microbatch_size": 16000,
            "training_mode": "joint",
        },
    }
    assert train_graph.resolve_options(_args(), saved).subgraph_size == 2000
    with pytest.raises(ValueError, match="subgraph_size"):
        train_graph.resolve_options(_args("--subgraph-size", "1000"), saved)
    legacy = {"model_id": "unit", "prefill_chunk": 16000, "config": {}}
    assert train_graph.resolve_options(_args(), legacy).subgraph_size is None
    with pytest.raises(ValueError, match="subgraph_size"):
        train_graph.resolve_options(_args("--subgraph-size", "1000"), legacy)


@pytest.mark.parametrize("option", ("--save-every", "--eval-every"))
def test_cadence_intervals_must_be_positive(option):
    with pytest.raises(ValueError, match="positive integer"):
        train_graph.resolve_options(_args(option, "0"))


def test_cadence_due_distinguishes_training_context_steps_from_epochs():
    assert train_graph.cadence_due(
        "steps", 2, train_steps=2, completed_epoch=False, contexts_per_epoch=34
    )
    assert not train_graph.cadence_due(
        "steps", 2, train_steps=1, completed_epoch=False, contexts_per_epoch=34
    )
    assert not train_graph.cadence_due(
        "epochs", 1, train_steps=34, completed_epoch=False, contexts_per_epoch=34
    )
    assert train_graph.cadence_due(
        "epochs", 1, train_steps=34, completed_epoch=True, contexts_per_epoch=34
    )


@pytest.mark.parametrize(
    ("mode", "result", "elapsed", "expected"),
    [
        (
            "joint",
            {
                "joint_loss": 0.5,
                "gate_loss": None,
                "graph_loss": None,
                "gradient_norm": 3.0,
                "gate_gradient_norm": 1.0,
                "mixer_gradient_norm": 2.0,
            },
            {"joint_forward_seconds": 6.0, "joint_backward_seconds": 3.0},
            {
                "train/bce": 0.5,
                "train/grad_norm": 3.0,
                "train/gate_grad_norm": 1.0,
                "train/mixer_grad_norm": 2.0,
                "timing/joint_forward_seconds_per_token": 2.0,
                "timing/joint_backward_seconds_per_token": 1.0,
            },
        ),
        (
            "two-phase",
            {
                "joint_loss": None,
                "gate_loss": 0.25,
                "graph_loss": 0.75,
                "gradient_norm": 3.0,
                "gate_gradient_norm": 1.0,
                "mixer_gradient_norm": 2.0,
            },
            {"gate_forward_seconds": 3.0, "graph_backward_seconds": 6.0},
            {
                "train/gate_bce": 0.25,
                "train/mixer_bce": 0.75,
                "train/grad_norm": 3.0,
                "train/gate_grad_norm": 1.0,
                "train/mixer_grad_norm": 2.0,
                "timing/gate_forward_seconds_per_token": 1.0,
                "timing/mixer_backward_seconds_per_token": 2.0,
            },
        ),
    ],
)
def test_context_wandb_metrics_use_compact_normalized_namespaces(
    monkeypatch, mode, result, elapsed, expected
):
    parameter = torch.nn.Parameter(torch.zeros(()))
    trainer = SimpleNamespace(
        scorer=SimpleNamespace(
            device=torch.device("cpu"),
            mixer=SimpleNamespace(alpha=torch.tensor([1.0, 3.0])),
        ),
        gate_optimizer=torch.optim.SGD([parameter], lr=0.01),
        mixer_optimizer=torch.optim.SGD([parameter], lr=0.02),
        timing=None,
        train_context=lambda *_args, **_kwargs: dict(result),
    )
    logged = {}
    run = SimpleNamespace(
        log=lambda metrics, *, step: logged.update(metrics=metrics, step=step)
    )
    monkeypatch.setattr(
        train_graph,
        "PhaseTiming",
        lambda *_: SimpleNamespace(resolve=lambda: elapsed),
    )
    _, metrics = train_graph.run_and_log_context(
        trainer,
        _example(),
        mode=mode,
        validation=False,
        run=run,
        step=7,
        fractional_epoch=0.5,
        cumulative_training_tokens=12,
    )
    assert metrics == {
        **expected,
        "train/mean_alpha": 2.0,
        "train/epoch": 0.5,
        "train/tokens": 12,
        "train/gate_learning_rate": 0.01,
        "train/mixer_learning_rate": 0.02,
    }
    assert logged == {"metrics": metrics, "step": 7}


def test_cursor_tracks_context_tokens_across_resume():
    first, completed_epoch = train_graph.advance_train_cursor(
        train_graph.initial_cursor(), token_count=7, contexts_per_epoch=2
    )
    assert not completed_epoch
    assert first["training_tokens"] == 7
    resumed, completed_epoch = train_graph.advance_train_cursor(
        first, token_count=11, contexts_per_epoch=2
    )
    assert completed_epoch
    assert resumed["training_tokens"] == 18
    assert train_graph.training_context_steps(resumed, 2) == 2


def test_removed_faiss_gin_and_b_init_options_are_not_accepted():
    parser = train_graph.build_parser()
    for option in ("--b-init", "--gin-depth", "--num-neighbors", "--knn-index"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "unit", option, "x"])


def test_normalized_checkpoint_configuration_excludes_cache_path():
    options = train_graph.resolve_options(
        _args("--teacher-cache-dir", "cache", "--adamw-eps", "1e-4", "--amsgrad")
    )
    assert options.teacher_cache_dir.name == "cache"
    scorer = SimpleNamespace(
        compute_dtype=torch.float32,
        gate_dim=1,
        gates=[SimpleNamespace(sink=1)],
        hidden_dim=2,
        num_layers=1,
        num_heads=1,
    )
    config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=scorer, options=options, query_groups=1
    )
    assert "teacher_cache_dir" not in config
    assert config["activation_order"] == "batchnorm-leaky-relu"
    assert config["adamw_eps"] == pytest.approx(1e-4)
    assert config["amsgrad"]


def test_normalized_checkpoint_configuration_preserves_train_context_count():
    options = train_graph.resolve_options(_args("--train-context-count", "50"))
    scorer = SimpleNamespace(
        compute_dtype=torch.float32,
        gate_dim=1,
        gates=[SimpleNamespace(sink=1)],
        hidden_dim=2,
        num_layers=1,
        num_heads=1,
    )
    config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=scorer, options=options, query_groups=1
    )

    assert config["train_context_count"] == 50


def test_checkpoint_configuration_stores_only_enabled_subgraph_size():
    scorer = SimpleNamespace(
        compute_dtype=torch.float32,
        gate_dim=1,
        gates=[SimpleNamespace(sink=1)],
        hidden_dim=2,
        num_layers=1,
        num_heads=1,
    )
    whole = train_graph.resolve_options(_args())
    chunked = train_graph.resolve_options(
        _args("--subgraph-size", "2", "--token-microbatch-size", "4")
    )
    whole_config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=scorer, options=whole, query_groups=1
    )
    chunked_config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=scorer, options=chunked, query_groups=1
    )
    assert "subgraph_size" not in whole_config
    assert chunked_config["subgraph_size"] == 2


def test_teacher_cache_atomic_creation_reuse_partial_and_mismatch_failures(tmp_path):
    first_path = train_graph._teacher_cache_path(tmp_path, ("fineweb_10k", 0))
    second_path = train_graph._teacher_cache_path(tmp_path, ("fineweb_10k", 1))
    assert first_path.name == "source-0.pt"
    first = _example(0)
    train_graph._save_teacher_cache_if_missing(
        first_path, first, model_id="unit", prefill_chunk=4
    )
    assert first_path.exists()
    loaded = train_graph._load_teacher_cache(
        first_path, key=("fineweb_10k", 0), model_id="unit", prefill_chunk=4
    )
    assert loaded.dataset_index == 0
    assert not second_path.exists()
    train_graph._save_teacher_cache_if_missing(
        second_path, _example(1), model_id="unit", prefill_chunk=4
    )
    with pytest.raises(FileExistsError):
        train_graph._save_teacher_cache_if_missing(
            first_path, first, model_id="unit", prefill_chunk=4
        )
    with pytest.raises(ValueError, match="incompatible"):
        train_graph._load_teacher_cache(
            first_path, key=("fineweb_10k", 0), model_id="other", prefill_chunk=4
        )


def test_corrupt_teacher_cache_fails_without_regeneration(tmp_path):
    path = train_graph._teacher_cache_path(tmp_path, ("fineweb_10k", 0))
    path.parent.mkdir()
    path.write_bytes(b"not a torch payload")
    with pytest.raises(ValueError, match="invalid or incompatible"):
        train_graph._load_teacher_cache(
            path, key=("fineweb_10k", 0), model_id="unit", prefill_chunk=4
        )


def test_run_training_reuses_hits_and_generates_only_missing_partial_cache(
    tmp_path, monkeypatch
):
    training_data = _training_data(
        (("fineweb_10k", 0), ("fineweb_10k", 1))
    )
    calls, released, teacher_refs = [], [], []

    class Teacher:
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=2,
        )
        tokenizer = None
        device = torch.device("cpu")
        dtype = torch.float32
        model = SimpleNamespace(name_or_path="unit")
        sys_prompt_ids = torch.tensor([[9]], dtype=torch.long)

    class Wrapper:
        def prefill_context(self, index, **kwargs):
            calls.append(index)
            return SimpleNamespace(
                start_idx=1,
                end_idx=4,
                hidden_cache=[torch.randn(1, 4, 2)],
                score=torch.rand(1, 1, 1, 4),
                prefill_ids=torch.tensor([[9, 1, 2, 3]], dtype=torch.long),
            )

    class Run:
        class Config:
            def update(self, *args, **kwargs):
                pass

        config = Config()
        id = None

        def finish(self, exit_code=None):
            pass

    trainer = SimpleNamespace(
        scorer=SimpleNamespace(device=torch.device("cpu")),
        gate_optimizer=None,
        mixer_optimizer=None,
        gate_scheduler=None,
        mixer_scheduler=None,
    )

    def build_teacher(*args, **kwargs):
        teacher = Teacher()
        teacher_refs.append(weakref.ref(teacher))
        return teacher

    monkeypatch.setattr(train_graph, "build_teacher", build_teacher)
    monkeypatch.setattr(
        train_graph,
        "_make_components",
        lambda teacher, options, resume, total_steps: (
            options,
            trainer.scorer,
            trainer,
            {},
        ),
    )
    monkeypatch.setattr(train_graph, "_initialize_wandb", lambda *args, **kwargs: Run())
    monkeypatch.setattr(train_graph, "save_checkpoint", lambda *args, **kwargs: None)

    def run_context(*args, **kwargs):
        released.append(teacher_refs[-1]() is None)
        return (
            {
                "validation_loss": None,
                "gate_loss": 0.0,
                "graph_loss": None,
                "joint_loss": 0.0,
            },
            {},
        )

    monkeypatch.setattr(train_graph, "run_and_log_context", run_context)
    cache_dir = tmp_path / "cache"
    args = _args(
        "--teacher-cache-dir",
        str(cache_dir),
        "--max-contexts",
        "1",
        "--eval-strategy",
        "steps",
        "--eval-every",
        "3",
        "--wandb-mode",
        "disabled",
    )
    train_graph.run_training(
        args,
        data_builder=lambda _count: training_data,
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == [0]
    calls.clear()
    train_graph.run_training(
        args,
        data_builder=lambda _count: training_data,
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == []

    partial_args = _args(
        "--teacher-cache-dir",
        str(cache_dir),
        "--max-contexts",
        "2",
        "--eval-strategy",
        "steps",
        "--eval-every",
        "3",
        "--wandb-mode",
        "disabled",
    )
    train_graph.run_training(
        partial_args,
        data_builder=lambda _count: training_data,
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == [1]

    calls.clear()
    train_graph.run_training(
        args,
        data_builder=lambda _count: training_data,
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == []
    assert released[-1]


@pytest.mark.parametrize(
    ("save_best_flag", "expected_saves"),
    [([], ["best", "last"]), (["--no-save-best"], ["last"])],
)
def test_cadence_evaluates_full_sweeps_without_validation_context_checkpoints(
    monkeypatch, tmp_path, save_best_flag, expected_saves
):
    class Teacher:
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=2,
        )
        tokenizer = None
        device = torch.device("cpu")
        dtype = torch.float32
        model = SimpleNamespace(name_or_path="unit")
        sys_prompt_ids = torch.tensor([[9]], dtype=torch.long)

    class Wrapper:
        def __init__(self, name):
            self.name = name

        def prefill_context(self, index, **kwargs):
            return SimpleNamespace(
                start_idx=1,
                end_idx=4,
                hidden_cache=[torch.randn(1, 4, 2)],
                score=torch.rand(1, 1, 1, 4),
                prefill_ids=torch.tensor([[9, 1, 2, 3]], dtype=torch.long),
            )

    class Run:
        class Config:
            def update(self, *args, **kwargs):
                pass

        config = Config()
        id = None

        def __init__(self):
            self.logged = []

        def log(self, metrics, *, step):
            self.logged.append((metrics, step))

        def finish(self, exit_code=None):
            pass

    class Progress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.postfixes = []
            self.descriptions = []
            self.closed = False

        def update(self, value):
            self.updates.append(value)

        def set_postfix(self, values):
            self.postfixes.append(values)

        def set_description(self, value):
            self.descriptions.append(value)

        def close(self):
            self.closed = True

    validation_means, calls, saves, training_progress = [], [], [], []
    validation_losses = iter((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    progress_bars = []
    trainer = SimpleNamespace(
        scorer=SimpleNamespace(device=torch.device("cpu")),
        gate_optimizer=None,
        mixer_optimizer=None,
        gate_scheduler=None,
        mixer_scheduler=None,
        step_validation=lambda loss: validation_means.append(loss),
    )
    run = Run()
    monkeypatch.setattr(train_graph, "build_teacher", lambda *args, **kwargs: Teacher())
    monkeypatch.setattr(
        train_graph,
        "_make_components",
        lambda teacher, options, resume, total_steps: (
            options, trainer.scorer, trainer, {}
        ),
    )
    monkeypatch.setattr(train_graph, "_initialize_wandb", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        train_graph,
        "save_checkpoint",
        lambda _dir, kind, **kwargs: saves.append((kind, dict(kwargs["data_cursor"]))),
    )

    def run_context(_trainer, example, *, validation, log_metrics=True, step, **kwargs):
        calls.append((example.dataset_name, example.dataset_index, validation, log_metrics, step))
        if not validation:
            training_progress.append(
                (kwargs["fractional_epoch"], kwargs["cumulative_training_tokens"])
            )
        return (
            {
                "validation_loss": next(validation_losses) if validation else None,
                "gate_loss": None,
                "graph_loss": None,
                "joint_loss": 0.0 if not validation else None,
            },
            {} if validation else {"train/bce": 0.0},
        )

    def progress_factory(**kwargs):
        progress = Progress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(train_graph, "run_and_log_context", run_context)
    training_data = _training_data(
        (("fineweb_10k", 0), ("fineweb_10k_cat", 1)),
        (
            ("fineweb_10k", 2),
            ("fineweb_10k", 3),
            ("fineweb_10k_cat", 4),
            ("fineweb_10k_cat", 5),
        ),
    )
    train_graph.run_training(
        _args(
            "--output-dir", str(tmp_path),
            "--max-contexts", "2",
            "--save-strategy", "steps", "--save-every", "3",
            "--eval-strategy", "steps", "--eval-every", "1",
            "--wandb-mode", "disabled",
            *save_best_flag,
        ),
        data_builder=lambda _count: training_data,
        wrapper_factory=lambda name, *args: Wrapper(name),
        progress_factory=progress_factory,
    )

    assert [call[:3] for call in calls if not call[2]] == [
        ("fineweb_10k", 0, False),
        ("fineweb_10k_cat", 1, False),
    ]
    assert [call[:3] for call in calls if call[2]] == [
        ("fineweb_10k", 2, True),
        ("fineweb_10k", 3, True),
        ("fineweb_10k_cat", 4, True),
        ("fineweb_10k_cat", 5, True),
        ("fineweb_10k", 2, True),
        ("fineweb_10k", 3, True),
        ("fineweb_10k_cat", 4, True),
        ("fineweb_10k_cat", 5, True),
    ]
    assert all(not call[3] for call in calls if call[2])
    assert [kind for kind, _ in saves] == expected_saves
    assert saves[-1][1]["best_validation_bce"] == pytest.approx(0.25)
    assert saves[-1][1]["training_tokens"] == 6
    assert validation_means == pytest.approx([0.25, 0.65])
    assert run.logged == [({"validation/bce": 0.25}, 1), ({"validation/bce": 0.65}, 3)]
    assert training_progress == [(0.5, 3), (1.0, 6)]
    assert len(progress_bars) == 1
    assert progress_bars[0].kwargs == {
        "total": 2,
        "initial": 0,
        "desc": "Training",
        "unit": "context",
        "position": 1,
    }
    assert progress_bars[0].updates == [1, 1]
    assert progress_bars[0].postfixes == [{"bce": 0.0}, {"bce": 0.0}]
    assert progress_bars[0].descriptions == [
        "Validating",
        "Training",
        "Validating",
        "Training",
    ]
    assert progress_bars[0].closed


def test_run_training_marks_wandb_failed_on_exception(monkeypatch):
    class Run:
        def __init__(self):
            self.exit_code = None

        def finish(self, exit_code=None):
            self.exit_code = exit_code

    run = Run()
    monkeypatch.setattr(train_graph, "_initialize_wandb", lambda *args, **kwargs: run)

    def fail_teacher(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(train_graph, "build_teacher", fail_teacher)
    with pytest.raises(RuntimeError, match="boom"):
        train_graph.run_training(_args("--wandb-mode", "disabled"))
    assert run.exit_code == 1


def test_wandb_run_id_is_saved_before_init_and_reused_after_restart(
    monkeypatch, tmp_path
):
    run_id_path = tmp_path / "wandb_run_id.txt"
    generated = []
    init_calls = []

    def generate_id():
        generated.append(True)
        return "stable12"

    def init(**kwargs):
        assert run_id_path.read_text(encoding="utf-8").strip() == "stable12"
        init_calls.append(kwargs)
        return SimpleNamespace(finish=lambda exit_code=None: None)

    wandb_module = SimpleNamespace(
        util=SimpleNamespace(generate_id=generate_id),
        login=lambda: True,
        init=init,
    )

    def stop_before_model_load(*args, **kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(train_graph, "build_teacher", stop_before_model_load)
    args = _args("--output-dir", str(tmp_path))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="stop"):
            train_graph.run_training(args, wandb_module=wandb_module)

    assert generated == [True]
    assert [call["id"] for call in init_calls] == ["stable12", "stable12"]
    assert [call["resume"] for call in init_calls] == ["allow", "allow"]


def test_teacher_example_from_kv_transfers_normal_hidden_storage_without_clone():
    hidden = [torch.randn(1, 5, 2)]
    kv = SimpleNamespace(
        start_idx=2,
        end_idx=5,
        hidden_cache=hidden,
        score=torch.rand(1, 1, 1, 5),
        prefill_ids=torch.arange(5).view(1, -1),
    )
    example = train_graph.teacher_example_from_kv(kv, "fineweb_10k", 0)
    assert example.hidden_by_layer[0].untyped_storage().data_ptr() == hidden[0].untyped_storage().data_ptr()
    assert example.sequence_length == 3


def test_normal_capture_converts_an_inference_tensor_to_a_normal_cpu_tensor():
    with torch.inference_mode():
        captured = torch.randn(2)
    normal = train_graph._normal_capture(captured)
    assert normal.device.type == "cpu"
    assert not normal.is_inference()
    assert not normal.requires_grad


def test_resume_configuration_does_not_require_the_teacher_cache_path():
    options = train_graph.resolve_options(_args("--teacher-cache-dir", "one"))
    saved = {
        "model_id": "unit",
        "compute_dtype": "float32",
        "gate_dim": 16,
        "gate_sink": 16,
        "hidden_dim": 2,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 32,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
        "graph_microbatch_size": 1,
        "training_mode": "joint",
        "token_microbatch_size": 1000,
        "gate_lr": 1e-4,
        "mixer_lr": 1e-3,
        "gate_lr_scheduler": None,
        "mixer_lr_scheduler": None,
        "freeze_gate": False,
        "train_context_count": 29,
    }
    assert "teacher_cache_dir" not in saved
    assert options.teacher_cache_dir.name == "one"
