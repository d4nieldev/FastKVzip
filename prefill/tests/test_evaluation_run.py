import json
import os
import subprocess
from pathlib import Path

import pytest

from results.evaluation_run import EvaluationRun, atomic_write_json


def _outputs(ratio, *, suffix="", full=None):
    return {
        "qa": [
            [
                [ratio, ratio + 0.01, 0.75],
                {
                    "pruned": f"pruned{suffix}",
                    "full__": full,
                    "answer": "answer",
                },
            ]
        ],
        "qa-1": [
            [
                [ratio, ratio + 0.01, 0.75],
                {
                    "pruned": f"pruned-1{suffix}",
                    "full__": None if full is None else f"{full}-1",
                    "answer": "answer-1",
                },
            ]
        ],
    }


def _open(results, checkpoint, *, mode="fail", window_size=4096, level="pair"):
    return EvaluationRun.open(
        results,
        "run",
        checkpoint_path=checkpoint,
        wandb_run_id="training-run",
        window_size=window_size,
        level=level,
        existing_results=mode,
    )


def test_manifest_is_minimal_and_checks_checkpoint_path_and_run_id(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    results = tmp_path / "results"

    with _open(results, checkpoint) as run:
        assert json.loads(run.manifest_path.read_text()) == {
            "checkpoint_path": str(checkpoint.resolve()),
            "wandb_run_id": "training-run",
            "window_size": 4096,
            "level": "pair",
        }
        assert {path.name for path in run.run_dir.iterdir()} == {
            "manifest.json",
            "outputs",
        }

    with pytest.raises(FileExistsError, match="already exists"):
        _open(results, checkpoint)
    with _open(results, checkpoint, mode="resume"):
        pass
    with pytest.raises(ValueError, match="window_size"):
        _open(results, checkpoint, mode="resume", window_size=1)

    copied = tmp_path / "copied.pt"
    copied.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match="checkpoint_path"):
        EvaluationRun.open(
            results,
            "run",
            checkpoint_path=copied,
            wandb_run_id="training-run",
            window_size=4096,
            level="pair",
            existing_results="resume",
        )

    with pytest.raises(ValueError, match="wandb_run_id"):
        EvaluationRun.open(
            results,
            "run",
            checkpoint_path=checkpoint,
            wandb_run_id="different-run",
            window_size=4096,
            level="pair",
            existing_results="resume",
        )

    checkpoint.write_bytes(b"different weights")
    with _open(results, checkpoint, mode="resume"):
        pass


def test_resume_creates_absent_run_and_overwrite_replaces_exact_run(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    results = tmp_path / "results"
    neighbor = results / "keep" / "file.txt"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text("keep")

    with _open(results, checkpoint, mode="resume") as run:
        run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        output_path = run.output_path("task", 0)
        assert output_path.is_file()

    with _open(results, checkpoint, mode="overwrite") as run:
        assert not output_path.exists()
        assert run.manifest_path.is_file()
    assert neighbor.read_text() == "keep"


def test_ratio_merge_and_full_answer_backfill_are_additive(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    with _open(tmp_path / "results", checkpoint) as run:
        first = run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        assert first.requested_ratios == (0.2,)
        assert first.answers == {"qa": "answer", "qa-1": "answer-1"}
        assert not first.has_full_answers
        assert json.loads(first.path.read_text()) == _outputs(0.2)

        merged = run.merge_example(
            "task",
            0,
            outputs=_outputs(0.3, suffix="-new", full="full"),
        )
        assert merged.requested_ratios == (0.2, 0.3)
        assert merged.full_answers == {"qa": "full", "qa-1": "full-1"}
        assert merged.has_full_answers
        assert all(entry[1]["full__"] == "full" for entry in merged.payload["qa"])

        with pytest.raises(ValueError, match="duplicate result"):
            run.merge_example(
                "task",
                0,
                outputs=_outputs(0.2),
            )

        backfilled = run.merge_example(
            "task",
            1,
            outputs=_outputs(0.2),
        )
        assert not backfilled.has_full_answers
        backfilled = run.merge_example(
            "task",
            1,
            full_answers={"qa": "later", "qa-1": "later-1"},
        )
        assert backfilled.full_answers == {"qa": "later", "qa-1": "later-1"}


def test_merge_rejects_duplicate_ratios_and_partial_formats(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    with _open(tmp_path / "results", checkpoint) as run:
        run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        duplicate = _outputs(0.2)
        with pytest.raises(ValueError, match="duplicate result"):
            run.merge_example(
                "task",
                0,
                outputs=duplicate,
            )
        with pytest.raises(ValueError, match="formats"):
            run.merge_example(
                "task",
                0,
                outputs={"qa": _outputs(0.3)["qa"]},
            )
        with pytest.raises(ValueError, match="full answers must exactly match"):
            run.merge_example(
                "task",
                0,
                full_answers={"qa": "missing qa-1"},
            )


def test_loading_rejects_invalid_json_duplicates_and_full_answer_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    with _open(tmp_path / "results", checkpoint) as run:
        result = run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        result.path.write_text("not json")
        with pytest.raises(ValueError, match="valid JSON"):
            run.load_example("task", 0)

        payload = _outputs(0.2)
        payload["qa"].append(_outputs(0.2)["qa"][0])
        atomic_write_json(result.path, payload)
        with pytest.raises(ValueError, match="duplicate requested ratio"):
            run.load_example("task", 0)

        payload = _outputs(0.2)
        payload["qa"].extend(_outputs(0.3, full="full")["qa"])
        atomic_write_json(result.path, payload)
        with pytest.raises(ValueError, match="full-cache answer changed"):
            run.load_example("task", 0)


def test_tasks_and_examples_form_a_union_on_disk(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    with _open(tmp_path / "results", checkpoint) as run:
        for task, index in (("task-1", 0), ("task-2", 1)):
            run.merge_example(
                task,
                index,
                outputs=_outputs(0.2),
            )
        assert [(item.task, item.example_index) for item in run.iter_examples()] == [
            ("task-1", 0),
            ("task-2", 1),
        ]


def test_noncanonical_output_filename_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    with _open(tmp_path / "results", checkpoint) as run:
        result = run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        duplicate = result.path.with_name("00.json")
        duplicate.write_bytes(result.path.read_bytes())
        with pytest.raises(ValueError, match="not canonical"):
            list(run.iter_examples())


def test_load_for_postprocessing_and_atomic_metrics(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    results = tmp_path / "results"
    with _open(results, checkpoint) as run:
        run.merge_example(
            "task",
            0,
            outputs=_outputs(0.2),
        )
        run_dir = run.run_dir

    with EvaluationRun.load(run_dir) as run:
        examples = list(run.iter_examples())
        assert [(item.task, item.example_index) for item in examples] == [("task", 0)]
        run.write_metrics({"tasks": {"task": {}}})
        assert json.loads(run.metrics_path.read_text()) == {"tasks": {"task": {}}}

        original = run.metrics_path.read_text()
        monkeypatch.setattr(
            "results.evaluation_run.os.replace",
            lambda *_: (_ for _ in ()).throw(OSError("stop")),
        )
        with pytest.raises(OSError, match="stop"):
            atomic_write_json(run.metrics_path, {"changed": True})
        assert run.metrics_path.read_text() == original
        assert not list(run.run_dir.glob(".metrics.json.*.tmp"))

    checkpoint.write_bytes(b"changed")
    with EvaluationRun.load(run_dir):
        pass
    checkpoint.unlink()
    with EvaluationRun.load(run_dir):
        pass


def test_evaluation_shell_scripts_parse_and_helper_dry_run(tmp_path):
    project = Path(__file__).resolve().parents[2]
    helper = project / "slurm" / "submit_eval_graph.sh"
    batch = project / "slurm" / "eval_graph.sbatch"
    subprocess.run(["bash", "-n", helper, batch], check=True)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "activate").write_text("", encoding="utf-8")
    run_name = f"pytest-{tmp_path.name}"
    completed = subprocess.run(
        [
            "bash",
            helper,
            run_name,
            "--gpu",
            "rtx_6000:1",
            "--time",
            "01:00:00",
            "--mem",
            "60G",
            "--graph-checkpoint",
            checkpoint,
            "--existing-results",
            "resume",
            "--log-to-wandb",
            "--wandb-project",
            "project",
            "--data",
            "squad",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "FASTKVZIP_VENV": str(environment)},
    )

    command = completed.stdout
    assert f"--job-name={run_name}" in command
    assert "%j-%x.log" in command
    assert "--existing-results resume" in command
    assert "--log-to-wandb --wandb-project project" in command
    assert "--data squad" in command
