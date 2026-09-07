import json

import pytest

from results.evaluation_run import EvaluationRun, atomic_write_json
from results.metric import evaluate_answer


@pytest.mark.parametrize("task", ["niah_multivalue", "vt", "cwe", "fwe"])
def test_ruler_recall_preserves_targets_and_does_not_normalize_punctuation(task):
    assert evaluate_answer(
        ["The answer is ALPHA; a-b\x00done"],
        [["alpha", "a-b", "ab"]],
        f"ruler_{task}_4k",
        "qa",
    ) == [pytest.approx(2 / 3)]


def test_ruler_qa_accepts_any_alias_without_article_or_number_normalization():
    assert evaluate_answer(
        ["NEW YORK"], [["NYC", "New York"]], "ruler_qa_1_4k", "qa"
    ) == [1]
    assert evaluate_answer(["one"], [["1"]], "ruler_qa_2_4k", "qa") == [0]
    assert evaluate_answer([""], [["a"]], "ruler_qa_2_4k", "qa") == [0]


def test_corrected_generation_cannot_resume_old_results_but_can_read_them(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    kwargs = dict(
        checkpoint_path=checkpoint,
        wandb_run_id="evaluation-only",
        window_size=0,
        level="pair",
    )
    with EvaluationRun.open(tmp_path, "run", **kwargs) as run:
        manifest = run.manifest
        assert manifest["generation_revision"] == 2
        manifest.pop("generation_revision")
        manifest.pop("ruler_prompt_mode")
        manifest.pop("dataset_revisions")
        atomic_write_json(run.manifest_path, manifest)
    assert EvaluationRun.load(tmp_path / "run").manifest["generation_revision"] == 1
    with pytest.raises(ValueError, match="generation_revision"):
        EvaluationRun.open(tmp_path, "run", existing_results="resume", **kwargs)


def test_ruler_prompt_and_pinned_revision_bind_resume_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    kwargs = dict(
        checkpoint_path=checkpoint,
        wandb_run_id=None,
        window_size=0,
        level="pair",
        ruler_prompt_mode="official",
        dataset_revisions={"ruler_4k": "a" * 40},
    )
    with EvaluationRun.open(tmp_path, "run", **kwargs) as run:
        assert (
            json.loads(run.manifest_path.read_text())["ruler_prompt_mode"] == "official"
        )
    with EvaluationRun.open(tmp_path, "run", existing_results="resume", **kwargs):
        pass
    with pytest.raises(ValueError, match="ruler_prompt_mode"):
        EvaluationRun.open(
            tmp_path,
            "run",
            existing_results="resume",
            **{**kwargs, "ruler_prompt_mode": "graphkv"},
        )
    with pytest.raises(ValueError, match="dataset_revisions"):
        EvaluationRun.open(
            tmp_path,
            "run",
            existing_results="resume",
            **{**kwargs, "dataset_revisions": {"ruler_4k": "b" * 40}},
        )


@pytest.mark.parametrize("module_name", ["eval_graph", "eval_graph_chunked"])
@pytest.mark.parametrize("override", [None, "new-evaluation-run"])
def test_both_evaluators_bind_destination_before_runtime_without_logging(
    monkeypatch, tmp_path, module_name, override
):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(module_name)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = SimpleNamespace(
        config={"num_layers": 1, "num_kv_heads": 1},
        graph_microbatch_size=1,
        payload={"wandb_run_id": "training-run"},
    )
    monkeypatch.setattr(
        module, "load_evaluation_checkpoint", lambda *a, **k: checkpoint
    )

    def stop_at_runtime(*args, **kwargs):
        raise RuntimeError("runtime boundary")

    monkeypatch.setattr(module, "build_evaluation_runtime", stop_at_runtime)
    argv = [
        "--graph-checkpoint",
        str(checkpoint_path),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    if override:
        argv += ["--wandb-run-id", override]
    args = module.build_parser().parse_args(argv)
    assert args.num is None
    assert not args.log_to_wandb
    assert args.ruler_prompt_mode == "graphkv"
    with pytest.raises(RuntimeError, match="runtime boundary"):
        module.run_evaluation(args)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["wandb_run_id"] == (override or "training-run")
    assert manifest["generation_revision"] == 2


@pytest.mark.parametrize("mode", ["graphkv", "official"])
def test_pilot_keeps_full_size_and_prompt_modes_have_distinct_result_keys(
    monkeypatch, tmp_path, mode
):
    from test_graph_eval import _run_fake_evaluation

    calls = []
    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        tasks=("ruler_vt_4k",),
        loader_calls=calls,
        extra_args=("--ruler-prompt-mode", mode),
        full_size=500,
    )
    assert calls[0][2]["n_data"] == 1
    task_name = "ruler_vt_4k" + ("_official" if mode == "official" else "")
    assert run.merges[0][0][0] == task_name
    assert run.finalizations[0][1:3] == (task_name, 500)
    assert bool(run.prefix_restores) == (mode == "graphkv")


def test_evaluation_without_limit_requests_complete_loader(monkeypatch, tmp_path):
    from test_graph_eval import _run_fake_evaluation

    calls = []
    _run_fake_evaluation(monkeypatch, tmp_path, loader_calls=calls, limit=None)
    assert calls[0][2]["n_data"] is None


def test_agentic_explicit_legacy_evaluation_keeps_its_bounded_loader_range(
    monkeypatch, tmp_path
):
    from test_graph_eval import _run_fake_evaluation

    calls = []
    _run_fake_evaluation(monkeypatch, tmp_path, tasks=("agentic",), loader_calls=calls)
    assert calls[0][2]["n_data"] == 100


def test_chunked_evaluation_rejects_deferred_teacher_before_compressed_prefill(
    monkeypatch, tmp_path
):
    import eval_graph_chunked

    monkeypatch.setattr(
        eval_graph_chunked,
        "load_evaluation_checkpoint",
        lambda *a, **k: pytest.fail("must reject before checkpoint/runtime"),
    )
    args = eval_graph_chunked.build_parser().parse_args(
        [
            "--graph-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--run-dir",
            str(tmp_path / "run"),
            "--data",
            "agentic",
        ]
    )
    with pytest.raises(ValueError, match="Agentic.*chunked"):
        eval_graph_chunked.run_evaluation(args)


def test_evaluator_and_result_parser_expand_the_same_complete_inventory():
    from eval import get_data_list
    from results.parse import get_data_list as result_data_list

    assert get_data_list is result_data_list
    assert len(get_data_list("all")) == 107


def test_ruler_macro_average_requires_all_tasks_at_each_ratio_and_separates_modes():
    from results.parse import ruler_macro_averages
    from data.benchmarks import RULER_TASKS

    tasks = {
        f"ruler_{task}_4k": {"ratios": {"0.2": {"score": i, "complete": True}}}
        for i, task in enumerate(RULER_TASKS)
    }
    tasks["ruler_vt_4k_official"] = {
        "ratios": {"0.2": {"score": 100, "complete": False}}
    }
    macros = ruler_macro_averages(tasks)
    assert macros["ruler_4k"]["0.2"] == {
        "score": 6.0,
        "task_count": 13,
        "expected_tasks": 13,
        "complete": True,
    }
    assert macros["ruler_4k_official"]["0.2"]["task_count"] == 1
    assert not macros["ruler_4k_official"]["0.2"]["complete"]
    tasks["ruler_vt_4k"]["ratios"]["0.2"]["complete"] = False
    assert not ruler_macro_averages(tasks)["ruler_4k"]["0.2"]["complete"]


def test_ruler_result_roundtrip_preserves_aliases_and_zero_full_baseline(tmp_path):
    from results.parse import build_run_metrics

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with EvaluationRun.open(
        tmp_path,
        "run",
        checkpoint_path=checkpoint,
        wandb_run_id=None,
        window_size=0,
        level="pair",
    ) as run:
        run.merge_example(
            "ruler_qa_1_4k",
            0,
            outputs={
                "qa": [
                    [
                        [0.2, 0.2, 0.0, 0.2],
                        {
                            "pruned": "New York",
                            "full__": "wrong",
                            "answer": ["NYC", "New York"],
                        },
                    ]
                ]
            },
        )
        metrics = build_run_metrics(run, dataset_sizes={"ruler_qa_1_4k": 500})
    task = metrics["tasks"]["ruler_qa_1_4k"]
    assert task["ratios"]["0.2"]["score"] == 100
    assert task["full_cache"]["score"] == 0
    assert not task["complete"]
    assert metrics["ruler_macro_averages"]["ruler_4k"]["0.2"]["task_count"] == 1
