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

        def finish(self):
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
