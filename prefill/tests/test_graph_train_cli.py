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


def test_cli_defaults_are_joint_implicit_mixer_defaults():
    options = train_graph.resolve_options(_args())
    assert options.mode == "joint"
    assert options.graph_dim == 32
    assert options.gram_normalization == "token-count"
    assert options.leaky_relu_slope == pytest.approx(0.01)
    assert options.alpha_init == pytest.approx(0.1)
    assert options.gate_lr == pytest.approx(1e-4)
    assert options.mixer_lr == pytest.approx(1e-3)
    assert options.graph_microbatch_size == "auto"
    assert options.teacher_cache_dir is None
    assert options.gate_scheduler is None
    assert options.mixer_scheduler is None
    assert options.save_strategy == "epochs"
    assert options.save_every == 1
    assert options.save_best
    assert options.eval_strategy == "epochs"
    assert options.eval_every == 1


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


@pytest.mark.parametrize("option", ("--save-every", "--eval-every"))
def test_cadence_intervals_must_be_positive(option):
    with pytest.raises(ValueError, match="positive integer"):
        train_graph.resolve_options(_args(option, "0"))


def test_cadence_due_distinguishes_training_context_steps_from_epochs():
    assert train_graph.cadence_due(
        "steps", 2, train_steps=2, completed_epoch=False
    )
    assert not train_graph.cadence_due(
        "steps", 2, train_steps=1, completed_epoch=False
    )
    assert not train_graph.cadence_due(
        "epochs", 1, train_steps=len(train_graph.TRAIN_KEYS), completed_epoch=False
    )
    assert train_graph.cadence_due(
        "epochs", 1, train_steps=len(train_graph.TRAIN_KEYS), completed_epoch=True
    )


@pytest.mark.parametrize(
    ("mode", "result", "elapsed", "expected"),
    [
        (
            "joint",
            {"joint_loss": 0.5, "gate_loss": None, "graph_loss": None},
            {"joint_forward_seconds": 6.0, "joint_backward_seconds": 3.0},
            {
                "train/bce": 0.5,
                "timing/joint_forward_seconds_per_token": 2.0,
                "timing/joint_backward_seconds_per_token": 1.0,
            },
        ),
        (
            "two-phase",
            {"joint_loss": None, "gate_loss": 0.25, "graph_loss": 0.75},
            {"gate_forward_seconds": 3.0, "graph_backward_seconds": 6.0},
            {
                "train/gate_bce": 0.25,
                "train/mixer_bce": 0.75,
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
        scorer=SimpleNamespace(device=torch.device("cpu")),
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
        trainer, _example(), mode=mode, validation=False, run=run, step=7
    )
    assert metrics == {
        **expected,
        "train/gate_learning_rate": 0.01,
        "train/mixer_learning_rate": 0.02,
    }
    assert logged == {"metrics": metrics, "step": 7}


def test_removed_faiss_gin_and_b_init_options_are_not_accepted():
    parser = train_graph.build_parser()
    for option in ("--b-init", "--gin-depth", "--num-neighbors", "--knn-index"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "unit", option, "x"])


def test_normalized_checkpoint_configuration_excludes_cache_path():
    options = train_graph.resolve_options(_args("--teacher-cache-dir", "cache"))
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


def test_teacher_cache_atomic_creation_reuse_partial_and_mismatch_failures(tmp_path):
    first_path = train_graph._teacher_cache_path(tmp_path, ("fineweb_10k", 0))
    second_path = train_graph._teacher_cache_path(tmp_path, ("fineweb_10k", 1))
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
    calls = []

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
    monkeypatch.setattr(train_graph, "build_teacher", lambda *args, **kwargs: Teacher())
    monkeypatch.setattr(
        train_graph,
        "_make_components",
        lambda teacher, options, resume: (
            options,
            trainer.scorer,
            trainer,
            {},
        ),
    )
    monkeypatch.setattr(train_graph, "_initialize_wandb", lambda *args, **kwargs: Run())
    monkeypatch.setattr(train_graph, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_graph,
        "run_and_log_context",
        lambda *args, **kwargs: (
            {
                "validation_loss": None,
                "gate_loss": 0.0,
                "graph_loss": None,
                "joint_loss": 0.0,
            },
            {},
        ),
    )
    cache_dir = tmp_path / "cache"
    args = _args(
        "--teacher-cache-dir",
        str(cache_dir),
        "--max-contexts",
        "1",
        "--wandb-mode",
        "disabled",
    )
    train_graph.run_training(
        args,
        dataset_loader=lambda *args: [],
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == [0]
    calls.clear()
    train_graph.run_training(
        args,
        dataset_loader=lambda *args: [],
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == []

    partial_args = _args(
        "--teacher-cache-dir",
        str(cache_dir),
        "--max-contexts",
        "2",
        "--wandb-mode",
        "disabled",
    )
    train_graph.run_training(
        partial_args,
        dataset_loader=lambda *args: [],
        wrapper_factory=lambda *args: Wrapper(),
    )
    assert calls == [1]


@pytest.mark.parametrize(
    ("save_best_flag", "expected_saves"),
    [([], ["best", "last"]), (["--no-save-best"], ["last"])],
)
def test_cadence_evaluates_full_sweeps_without_validation_context_checkpoints(
    monkeypatch, tmp_path, save_best_flag, expected_saves
):
    monkeypatch.setattr(train_graph, "TRAIN_KEYS", (("train", 0), ("train", 1)))
    monkeypatch.setattr(
        train_graph, "VALIDATION_KEYS", (("validation", 0), ("validation", 1))
    )

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

    validation_means, calls, saves = [], [], []
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
        lambda teacher, options, resume: (options, trainer.scorer, trainer, {}),
    )
    monkeypatch.setattr(train_graph, "_initialize_wandb", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        train_graph,
        "save_checkpoint",
        lambda _dir, kind, **kwargs: saves.append((kind, dict(kwargs["data_cursor"]))),
    )

    def run_context(_trainer, example, *, validation, log_metrics=True, step, **kwargs):
        calls.append((example.dataset_name, example.dataset_index, validation, log_metrics, step))
        return (
            {
                "validation_loss": 0.25 if validation else None,
                "gate_loss": None,
                "graph_loss": None,
                "joint_loss": 0.0 if not validation else None,
            },
            {},
        )

    monkeypatch.setattr(train_graph, "run_and_log_context", run_context)
    train_graph.run_training(
        _args(
            "--output-dir", str(tmp_path),
            "--max-contexts", "2",
            "--save-strategy", "steps", "--save-every", "3",
            "--eval-strategy", "steps", "--eval-every", "1",
            "--wandb-mode", "disabled",
            *save_best_flag,
        ),
        dataset_loader=lambda *args: [],
        wrapper_factory=lambda name, *args: Wrapper(name),
    )

    assert [call[:3] for call in calls if not call[2]] == [("train", 0, False), ("train", 1, False)]
    assert [call[:3] for call in calls if call[2]] == [
        ("validation", 0, True),
        ("validation", 1, True),
        ("validation", 0, True),
        ("validation", 1, True),
    ]
    assert all(not call[3] for call in calls if call[2])
    assert [kind for kind, _ in saves] == expected_saves
    assert saves[-1][1]["best_validation_bce"] == pytest.approx(0.25)
    assert validation_means == [0.25, 0.25]
    assert run.logged == [
        ({"validation/bce": 0.25}, 1),
        ({"validation/bce": 0.25}, 3),
    ]


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
    }
    assert "teacher_cache_dir" not in saved
    assert options.teacher_cache_dir.name == "one"
