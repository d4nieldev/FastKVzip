import json

import pytest

from results.evaluation_run import EvaluationRun


def _open(tmp_path, *, mode="fail"):
    checkpoint = tmp_path / "checkpoint.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"weights")
    return EvaluationRun.open(
        tmp_path,
        "run",
        checkpoint_path=checkpoint,
        wandb_run_id=None,
        window_size=0,
        level="pair-head",
        existing_results=mode,
    )


def test_dataset_size_is_durable_before_any_example_and_stable_on_resume(tmp_path):
    with _open(tmp_path) as run:
        assert run.dataset_sizes == {}
        run.record_dataset_size("gsm", 317)
        run_dir = run.run_dir
        assert list(run.iter_examples()) == []

    with EvaluationRun.load(run_dir) as loaded:
        assert loaded.dataset_sizes == {"gsm": 317}
    with _open(tmp_path, mode="resume") as resumed:
        resumed.record_dataset_size("gsm", 317)
        resumed.record_dataset_size("ruler_qa_1_4k", 500)
        with pytest.raises(ValueError, match="dataset size changed.*gsm"):
            resumed.record_dataset_size("gsm", 100)
        assert resumed.dataset_sizes == {"gsm": 317, "ruler_qa_1_4k": 500}


def test_failed_dataset_size_write_preserves_previous_record(tmp_path, monkeypatch):
    with _open(tmp_path) as run:
        run.record_dataset_size("gsm", 317)
        monkeypatch.setattr(
            "results.evaluation_run.os.replace",
            lambda *_a: (_ for _ in ()).throw(OSError("interrupted")),
        )
        with pytest.raises(OSError, match="interrupted"):
            run.record_dataset_size("ruler_qa_1_4k", 500)
        assert run.dataset_sizes == {"gsm": 317}
        assert not list(run.run_dir.glob(".datasets.json.*.tmp"))


@pytest.mark.parametrize("size", [-1, True, "317"])
def test_invalid_dataset_size_is_rejected_when_recording_or_loading(tmp_path, size):
    with _open(tmp_path) as run:
        with pytest.raises(ValueError, match="dataset size"):
            run.record_dataset_size("gsm", size)
        run.datasets_path.write_text(json.dumps({"gsm": size}))
        with pytest.raises(ValueError, match="dataset sizes"):
            _ = run.dataset_sizes


@pytest.mark.parametrize("previous_size", [None, 1])
def test_standalone_partial_gsm_parse_uses_recorded_size_without_loading_data(
    tmp_path, monkeypatch, previous_size
):
    import datasets
    from results import parse

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_a, **_k: pytest.fail("unexpected dataset download"),
    )
    with _open(tmp_path) as run:
        run.record_dataset_size("gsm", 317)
        run.merge_example(
            "gsm",
            0,
            outputs={
                "qa": [
                    [
                        [0.2, 0.2, 0.0],
                        {
                            "pruned": "The answer is 42",
                            "full__": None,
                            "answer": "#### 42",
                        },
                    ]
                ],
            },
        )
        if previous_size is not None:
            run.write_metrics({"tasks": {"gsm": {"dataset_size": previous_size}}})
        run_dir = run.run_dir

    parse.main(["--run-dir", str(run_dir)])

    result = json.loads((run_dir / "metrics.json").read_text())["tasks"]["gsm"]
    assert result["dataset_size"] == 317
    assert result["example_count"] == 1
    assert not result["complete"]
    assert result["ratios"]["0.2"]["score"] == 100.0


@pytest.mark.parametrize("legacy", [False, True])
def test_only_legacy_gsm_runs_can_fall_back_to_the_old_fixed_limit(tmp_path, legacy):
    from results import parse

    with _open(tmp_path) as run:
        run.merge_example(
            "gsm",
            0,
            outputs={
                "qa": [
                    [
                        [0.2, 0.2, 0.0],
                        {"pruned": "42", "full__": None, "answer": "#### 42"},
                    ]
                ],
            },
        )
        if legacy:
            manifest = run.manifest
            for key in (
                "generation_revision",
                "ruler_prompt_mode",
                "dataset_revisions",
            ):
                manifest.pop(key)
            run.manifest_path.write_text(json.dumps(manifest))
        run_dir = run.run_dir

    if legacy:
        parse.main(["--run-dir", str(run_dir)])
        result = json.loads((run_dir / "metrics.json").read_text())["tasks"]["gsm"]
        assert result["dataset_size"] == 100
    else:
        with pytest.raises(ValueError, match="GSM.*tokenizer.*datasets.json"):
            parse.main(["--run-dir", str(run_dir)])


@pytest.mark.parametrize("task", ["ruler_qa_1_4k", "ruler_qa_1_4k_official", "squad"])
def test_known_benchmark_sizes_can_be_recovered_without_finished_metrics(
    tmp_path, monkeypatch, task
):
    from data import load
    from results import parse

    monkeypatch.setattr(
        load,
        "load_dataset",
        lambda *_a, **_k: [
            {
                "context": f"context {i}",
                "question": "Where?",
                "answers": {"text": ["yes"]},
            }
            for i in range(103)
        ],
    )
    with _open(tmp_path) as run:
        run.merge_example(
            task,
            0,
            outputs={
                "qa": [
                    [
                        [0.2, 0.2, 0.0],
                        {
                            "pruned": "yes",
                            "full__": None,
                            "answer": ["yes"] if task.startswith("ruler_") else "yes",
                        },
                    ]
                ],
            },
        )
        run_dir = run.run_dir

    parse.main(["--run-dir", str(run_dir)])

    result = json.loads((run_dir / "metrics.json").read_text())["tasks"][task]
    assert result["dataset_size"] == (103 if task == "squad" else 500)
    assert not result["complete"]
