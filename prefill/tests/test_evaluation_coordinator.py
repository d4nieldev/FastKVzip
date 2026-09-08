import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from results.evaluation_run import EvaluationRun, atomic_write_json
from data.benchmarks import parse_ruler_name
from data.ruler import RULER_REVISIONS


class Wandb:
    """Only the external W&B boundary is replaced; uploads use the real code."""

    def __init__(self):
        self.history, self.api_paths, self.init_calls = [], [], []
        self.state = "finished"

    def Api(self):
        return SimpleNamespace(run=self.remote, default_entity="team")

    def remote(self, path):
        self.api_paths.append(path)
        return self

    def scan_history(self, *, keys, page_size):
        return (row for row in self.history if all(key in row for key in keys))

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.state = "running"
        return self

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values):
        self.history.append(dict(values))

    def finish(self, *, exit_code):
        self.state = "finished"


def _worker(tmp_path, name, task="scbench_kv", *, complete=True):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    revisions = {}
    size = 100
    if task.startswith("ruler_"):
        _, length = parse_ruler_name(task.removesuffix("_official"))
        revisions[f"ruler_{length}"] = RULER_REVISIONS[length]
        size = 500
    with EvaluationRun.open(
        tmp_path,
        name,
        checkpoint_path=checkpoint,
        wandb_run_id="evaluation-run",
        window_size=0,
        level="pair",
        prefill_mode="post-prefill",
        dataset_revisions=revisions,
    ) as run:
        run.record_dataset_size(task, size)
        metrics = {
            "tasks": {
                task: {
                    "complete": complete,
                    "example_count": size if complete else 1,
                    "dataset_size": size,
                    "full_cache": {
                        "score": 80.0,
                        "complete": complete,
                        "example_count": size if complete else 1,
                    },
                    "ratios": {
                        str(ratio): {
                            "score": 80.0 if ratio == 1 else 60.0,
                            "complete": complete,
                            "example_count": size if complete else 1,
                            "dataset_size": size,
                            "actual_retention": ratio,
                        }
                        for ratio in (1.0, 0.75, 0.5, 0.4, 0.3, 0.2)
                    },
                }
            }
        }
        atomic_write_json(run.metrics_path, metrics)
        return run.run_dir


def _upload(paths, wandb):
    from results.coordinator import upload_completed_runs

    return upload_completed_runs(
        paths,
        wandb_run_id="evaluation-run",
        project="existing-project",
        entity="team",
        wandb_module=wandb,
    )


def test_one_finished_benchmark_uploads_without_waiting_for_the_rest_and_retry_is_idempotent(
    tmp_path,
):
    worker = _worker(tmp_path, "finished")
    before = {path: path.read_bytes() for path in worker.rglob("*") if path.is_file()}
    wandb = Wandb()
    result = _upload([worker, tmp_path / "not-started"], wandb)
    assert result["uploaded_points"] == 6
    assert result["coverage"]["completed"] == 1
    assert result["coverage"]["expected"] == 107
    assert len(result["coverage"]["missing"]) == 106
    assert wandb.api_paths == ["team/existing-project/evaluation-run"]
    assert {row["test/retention_ratio"] for row in wandb.history} == {
        1,
        0.75,
        0.5,
        0.4,
        0.3,
        0.2,
    }
    assert all("test/scbench_kv" in row for row in wandb.history)
    assert _upload([worker], wandb)["uploaded_points"] == 0
    assert len(wandb.history) == 6
    assert {
        path: path.read_bytes() for path in worker.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("wandb_run_id", "original-training-run"),
        ("window_size", 0.02),
        ("window_revision", 0),
        ("level", "pair-head"),
        ("prefill_mode", "chunked"),
        ("ruler_prompt_mode", "official"),
        ("generation_revision", 1),
    ],
)
def test_every_manifest_is_validated_before_any_upload_even_when_later_worker_is_partial(
    tmp_path, field, value
):
    good = _worker(tmp_path, "first-good")
    bad = _worker(tmp_path, "later-bad", "gsm", complete=False)
    manifest_path = bad / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    atomic_write_json(manifest_path, manifest)
    wandb = Wandb()
    with pytest.raises(ValueError, match=field):
        _upload([good, bad], wandb)
    assert wandb.api_paths == wandb.init_calls == wandb.history == []


def test_old_eight_key_window_zero_manifest_blocks_all_uploads(tmp_path):
    good = _worker(tmp_path, "first-good")
    old = _worker(tmp_path, "old-window-zero", "gsm", complete=False)
    manifest_path = old / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("window_revision", None)
    assert len(manifest) == 8
    atomic_write_json(manifest_path, manifest)
    wandb = Wandb()
    with pytest.raises(ValueError, match="window_revision"):
        _upload([good, old], wandb)
    assert wandb.api_paths == wandb.init_calls == wandb.history == []


@pytest.mark.parametrize("task", ["agentic", "unknown", "ruler_qa_1_4k_official"])
def test_non_production_tasks_are_rejected_before_any_remote_access(tmp_path, task):
    good = _worker(tmp_path, "first-good")
    other = _worker(tmp_path, "other", task)
    wandb = Wandb()
    with pytest.raises(ValueError, match="task"):
        _upload([good, other], wandb)
    assert wandb.api_paths == wandb.init_calls == []


@pytest.mark.parametrize(
    "incomplete",
    ["pilot", "missing-ratio", "no-full-cache", "partial-ratio", "inconsistent-count"],
)
def test_incomplete_benchmarks_and_pilots_never_access_wandb(tmp_path, incomplete):
    worker = _worker(tmp_path, "pilot", complete=incomplete != "pilot")
    metrics_path = worker / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    task = metrics["tasks"]["scbench_kv"]
    if incomplete == "missing-ratio":
        del task["ratios"]["0.4"]
    elif incomplete == "no-full-cache":
        task["full_cache"]["complete"] = False
    elif incomplete == "partial-ratio":
        task["ratios"]["0.5"]["complete"] = False
    elif incomplete == "inconsistent-count":
        task["ratios"]["0.5"]["example_count"] = 1
    atomic_write_json(metrics_path, metrics)
    wandb = Wandb()
    result = _upload([worker], wandb)
    assert result["uploaded_points"] == result["coverage"]["completed"] == 0
    assert wandb.api_paths == wandb.init_calls == wandb.history == []


def test_identical_completed_copies_deduplicate_and_conflicting_copies_fail_before_upload(
    tmp_path,
):
    first = _worker(tmp_path, "first-copy")
    second = _worker(tmp_path, "second-copy")
    wandb = Wandb()
    summary = _upload([first, second], wandb)
    assert summary["uploaded_points"] == 6
    assert summary["coverage"]["completed"] == 1

    metrics_path = second / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["tasks"]["scbench_kv"]["ratios"]["0.2"]["score"] = 50.0
    atomic_write_json(metrics_path, metrics)
    wandb = Wandb()
    with pytest.raises(ValueError, match="conflicting"):
        _upload([first, second], wandb)
    assert wandb.api_paths == wandb.init_calls == []


def test_existing_remote_conflicts_and_running_destination_use_existing_upload_guards(
    tmp_path,
):
    worker = _worker(tmp_path, "worker")
    wandb = Wandb()
    wandb.history.append({"test/retention_ratio": 0.2, "test/scbench_kv": 1.0})
    with pytest.raises(ValueError, match="conflicts"):
        _upload([worker], wandb)
    assert wandb.init_calls == []
    wandb.state = "running"
    with pytest.raises(ValueError, match="not finished"):
        _upload([worker], wandb)
    assert wandb.init_calls == []


def test_ruler_summary_reports_incomplete_length_coverage_without_claiming_all_tasks(
    tmp_path,
):
    first = _worker(tmp_path, "niah", "ruler_niah_single_1_4k")
    second = _worker(tmp_path, "qa", "ruler_qa_1_4k")
    summary = _upload([first, second], Wandb())
    assert summary["coverage"]["completed"] == 2
    assert summary["ruler_macro_averages"]["ruler_4k"]["0.2"] == {
        "score": 60.0,
        "task_count": 2,
        "expected_tasks": 13,
        "complete": False,
    }


def test_cli_partial_workers_prints_summary_without_network_or_worker_writes(tmp_path):
    worker = _worker(tmp_path, "pilot", complete=False)
    before = {path: path.read_bytes() for path in worker.rglob("*") if path.is_file()}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "results.coordinator",
            str(worker),
            "--wandb-run-id",
            "evaluation-run",
            "--wandb-project",
            "existing-project",
            "--wandb-entity",
            "team",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["coverage"]["completed"] == 0
    assert {
        path: path.read_bytes() for path in worker.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("score", ["invalid-score", float("nan")])
def test_invalid_completed_metric_values_fail_before_remote_access(tmp_path, score):
    worker = _worker(tmp_path, "invalid")
    metrics_path = worker / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["tasks"]["scbench_kv"]["ratios"]["0.2"]["score"] = score
    metrics_path.write_text(json.dumps(metrics))
    wandb = Wandb()
    with pytest.raises(ValueError, match="metric"):
        _upload([worker], wandb)
    assert wandb.api_paths == wandb.init_calls == []


def test_conflicting_dataset_revisions_fail_before_remote_access(tmp_path):
    first = _worker(tmp_path, "first", "ruler_niah_single_1_4k")
    second = _worker(tmp_path, "second", "ruler_qa_1_4k")
    for path, revision in ((first, "a" * 40), (second, "b" * 40)):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset_revisions"] = {"ruler_4k": revision}
        atomic_write_json(manifest_path, manifest)
    wandb = Wandb()
    with pytest.raises(ValueError, match="revision"):
        _upload([first, second], wandb)
    assert wandb.api_paths == wandb.init_calls == []


@pytest.mark.parametrize("task", ["scbench_kv", "ruler_qa_1_4k"])
def test_false_complete_metrics_cannot_shrink_the_recorded_benchmark(tmp_path, task):
    worker = _worker(tmp_path, "shrunk", task)
    metrics_path = worker / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    values = metrics["tasks"][task]
    values["dataset_size"] = values["example_count"] = 1
    values["full_cache"]["example_count"] = 1
    for ratio in values["ratios"].values():
        ratio["dataset_size"] = ratio["example_count"] = 1
    atomic_write_json(metrics_path, metrics)
    wandb = Wandb()
    with pytest.raises(ValueError, match="size"):
        _upload([worker], wandb)
    assert wandb.api_paths == wandb.init_calls == []


def test_ruler_must_use_all_500_rows_even_if_local_size_record_is_wrong(tmp_path):
    worker = _worker(tmp_path, "wrong-size", "ruler_qa_1_4k")
    run = EvaluationRun.load(worker)
    atomic_write_json(run.datasets_path, {"ruler_qa_1_4k": 1})
    wandb = Wandb()
    with pytest.raises(ValueError, match="size"):
        _upload([worker], wandb)
    assert wandb.api_paths == wandb.init_calls == []
