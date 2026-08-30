import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from results import metric, parse
from results.evaluation_run import EvaluationRun


def _write_example(run_dir, task, index, ratios, *, full=None):
    path = run_dir / "outputs" / task / f"{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for requested, (prediction, actual) in ratios.items():
        entries.append(
            [
                [requested, actual, 0.0],
                {"pruned": str(prediction), "full__": full, "answer": "unused"},
            ]
        )
    path.write_text(
        json.dumps(
            {"qa": entries}
        ),
        encoding="utf-8",
    )


def _score(predictions, _answers, _task, _fmt, *, subtask=None):
    del subtask
    return [float(prediction) for prediction in predictions]


def _new_run(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return EvaluationRun.open(
        tmp_path,
        "run",
        checkpoint_path=checkpoint,
        wandb_run_id="training-run",
        window_size=4096,
        level="pair",
    )


def test_answer_metrics_do_not_print_per_prediction_progress(monkeypatch, capsys):
    monkeypatch.setattr(metric, "rouge_score", lambda *_args: 0.5)

    assert metric.evaluate_answer(["answer"], ["answer"], "squad", "qa") == [1]
    assert metric.evaluate_answer(["summary"], ["gold"], "task", "summary") == [0.5]
    assert capsys.readouterr().out == ""


def test_run_metrics_include_coverage_relative_and_actual_retention(tmp_path):
    with _new_run(tmp_path) as run:
        _write_example(run.run_dir, "squad", 0, {0.2: (0.5, 0.25)}, full="1.0")
        _write_example(run.run_dir, "squad", 1, {0.2: (0.75, 0.35)}, full="0.5")
        metrics = parse.build_run_metrics(
            run,
            dataset_sizes={"squad": 2},
            evaluate=_score,
            supplementary_loader=lambda _task: ([], []),
        )

    task = metrics["tasks"]["squad"]
    assert task["complete"]
    assert task["full_cache"] == {
        "score": 75.0,
        "example_count": 2,
        "complete": True,
    }
    assert task["ratios"]["0.2"] == {
        "score": 62.5,
        "relative": pytest.approx(62.5 / 75.0 * 100),
        "actual_retention": pytest.approx(0.3),
        "example_count": 2,
        "dataset_size": 2,
        "complete": True,
    }
    assert task["ratios"]["1.0"]["score"] == 75.0
    assert task["ratios"]["1.0"]["relative"] == 100.0
    assert metrics["average_relative_performance"]["0.2"] == pytest.approx(
        62.5 / 75.0 * 100
    )
    points = parse._wandb_points(metrics)
    assert points[("test/squad", "1.0")] == 75.0
    assert points[("test/squad-relative", "1.0")] == 100.0
    assert points[("test/squad-actual-retention", "1.0")] == 1.0


def test_partial_full_cache_omits_relative_and_partial_ratio_blocks_wandb(tmp_path):
    with _new_run(tmp_path) as run:
        _write_example(
            run.run_dir,
            "squad",
            0,
            {0.2: (0.5, 0.25), 0.3: (0.4, 0.35)},
            full="1.0",
        )
        _write_example(run.run_dir, "squad", 1, {0.2: (0.75, 0.35)})
        metrics = parse.build_run_metrics(
            run,
            dataset_sizes={"squad": 2},
            evaluate=_score,
            supplementary_loader=lambda _task: ([], []),
        )

    task = metrics["tasks"]["squad"]
    assert task["complete"]
    assert not task["full_cache"]["complete"]
    assert "relative" not in task["ratios"]["0.2"]
    assert task["ratios"]["0.3"]["example_count"] == 1
    assert not task["ratios"]["0.3"]["complete"]
    with pytest.raises(ValueError, match="complete ratio coverage"):
        parse._require_full_benchmarks(metrics)


def test_zero_full_cache_baseline_omits_relative_and_full_point(tmp_path):
    with _new_run(tmp_path) as run:
        _write_example(run.run_dir, "squad", 0, {0.2: (0.5, 0.25)}, full="0.0")
        metrics = parse.build_run_metrics(
            run,
            dataset_sizes={"squad": 1},
            evaluate=_score,
            supplementary_loader=lambda _task: ([], []),
        )

    task = metrics["tasks"]["squad"]
    assert task["full_cache"] == {
        "score": 0.0,
        "example_count": 1,
        "complete": True,
    }
    assert "relative" not in task["ratios"]["0.2"]
    assert "1.0" not in task["ratios"]


class _PublicRun:
    state = "finished"

    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, *, keys, page_size):
        assert page_size == 10000
        for row in self.rows:
            if all(key in row for key in keys):
                yield {key: row[key] for key in keys}


class _LiveRun:
    def __init__(self):
        self.definitions = []
        self.logged = []
        self.finished = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values):
        self.logged.append(values)

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class _Wandb:
    def __init__(self, rows):
        self.public = _PublicRun(rows)
        self.live = _LiveRun()
        self.init_calls = []
        self.api_paths = []

    def Api(self):
        def load(path):
            self.api_paths.append(path)
            return self.public

        return SimpleNamespace(run=load, default_entity="default-entity")

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.live


def _complete_metrics():
    return {
        "tasks": {
            "squad": {
                "complete": True,
                "ratios": {
                    "0.2": {
                        "score": 50.0,
                        "relative": 50.0,
                        "actual_retention": 0.25,
                        "complete": True,
                    }
                },
            }
        }
    }


def test_wandb_upload_skips_matches_and_appends_only_missing_curves():
    wandb = _Wandb(
        [{"test/retention_ratio": 0.2, "test/squad": 50.0}]
    )
    uploaded = parse.upload_run_metrics(
        _complete_metrics(),
        {"wandb_run_id": "training-run"},
        project="project",
        entity="entity",
        wandb_module=wandb,
    )

    assert uploaded == 2
    assert wandb.api_paths == ["entity/project/training-run"]
    assert wandb.init_calls[0]["id"] == "training-run"
    assert wandb.init_calls[0]["resume"] == "must"
    settings = dict(wandb.init_calls[0]["settings"])
    root_dir = settings.pop("root_dir")
    assert not Path(root_dir).exists()
    assert settings == {
        "x_disable_stats": True,
        "x_disable_meta": True,
    }
    assert wandb.live.logged == [
        {
            "test/retention_ratio": 0.2,
            "test/squad-actual-retention": 0.25,
            "test/squad-relative": 50.0,
        }
    ]
    assert wandb.live.finished == [0]


def test_wandb_conflict_fails_before_resuming_training_run():
    wandb = _Wandb(
        [{"test/retention_ratio": 0.2, "test/squad": 49.0}]
    )
    with pytest.raises(ValueError, match="conflicts with local"):
        parse.upload_run_metrics(
            _complete_metrics(),
            {"wandb_run_id": "training-run"},
            project="project",
            wandb_module=wandb,
        )
    assert wandb.init_calls == []


def test_wandb_retry_repairs_axes_without_relogging_matching_points():
    wandb = _Wandb(
        [
            {
                "test/retention_ratio": 0.2,
                "test/squad": 50.0,
                "test/squad-relative": 50.0,
                "test/squad-actual-retention": 0.25,
            }
        ]
    )
    uploaded = parse.upload_run_metrics(
        _complete_metrics(),
        {"wandb_run_id": "training-run"},
        project="project",
        wandb_module=wandb,
    )
    assert uploaded == 0
    assert len(wandb.init_calls) == 1
    assert wandb.live.logged == []
    assert wandb.live.definitions == [
        (("test/retention_ratio",), {"hidden": True}),
        (
            ("test/*",),
            {
                "step_metric": "test/retention_ratio",
                "step_sync": True,
                "overwrite": True,
            },
        )
    ]


def test_wandb_upload_ignores_sparse_history_rows():
    wandb = _Wandb(
        [
            {"test/retention_ratio": None, "test/squad": None},
            {"test/retention_ratio": 0.2, "test/squad": 50.0},
        ]
    )

    uploaded = parse.upload_run_metrics(
        _complete_metrics(),
        {"wandb_run_id": "training-run"},
        project="project",
        wandb_module=wandb,
    )

    assert uploaded == 2


def test_wandb_history_for_an_unselected_ratio_is_preserved():
    wandb = _Wandb(
        [
            {
                "test/retention_ratio": 0.2,
                "test/squad": 50.0,
                "test/squad-relative": 50.0,
                "test/squad-actual-retention": 0.25,
            },
            {
                "test/retention_ratio": 0.3,
                "test/squad": 60.0,
                "test/squad-relative": 60.0,
                "test/squad-actual-retention": 0.35,
            },
        ]
    )
    uploaded = parse.upload_run_metrics(
        _complete_metrics(),
        {"wandb_run_id": "training-run"},
        project="project",
        wandb_module=wandb,
    )

    assert uploaded == 0
    assert wandb.live.logged == []


def test_matching_duplicate_wandb_points_are_ignored():
    wandb = _Wandb(
        [
            {"test/retention_ratio": 0.2, "test/squad": 50.0},
            {"test/retention_ratio": 0.2, "test/squad": 50.0},
        ]
    )
    uploaded = parse.upload_run_metrics(
        _complete_metrics(),
        {"wandb_run_id": "training-run"},
        project="project",
        wandb_module=wandb,
    )
    assert uploaded == 2


def test_conflicting_duplicate_wandb_points_fail_before_resume():
    wandb = _Wandb(
        [
            {"test/retention_ratio": 0.2, "test/squad": 50.0},
            {"test/retention_ratio": 0.2, "test/squad": 49.0},
        ]
    )
    with pytest.raises(ValueError, match="conflicting W&B metric points"):
        parse.upload_run_metrics(
            _complete_metrics(),
            {"wandb_run_id": "training-run"},
            project="project",
            wandb_module=wandb,
        )
    assert wandb.init_calls == []


def test_wandb_upload_requires_a_finished_training_run():
    wandb = _Wandb([])
    wandb.public.state = "running"
    with pytest.raises(ValueError, match="not finished"):
        parse.upload_run_metrics(
            _complete_metrics(),
            {"wandb_run_id": "training-run"},
            project="project",
            wandb_module=wandb,
        )
    assert wandb.init_calls == []


def test_wandb_upload_failure_keeps_training_run_finished():
    wandb = _Wandb([])

    def fail(_values):
        raise RuntimeError("upload failed")

    wandb.live.log = fail
    with pytest.raises(RuntimeError, match="upload failed"):
        parse.upload_run_metrics(
            _complete_metrics(),
            {"wandb_run_id": "training-run"},
            project="project",
            wandb_module=wandb,
        )
    assert wandb.live.finished == [0]


def test_run_postprocessing_rebuilds_metrics_without_touching_wandb(
    tmp_path, monkeypatch
):
    with _new_run(tmp_path) as run:
        _write_example(run.run_dir, "squad", 0, {0.2: (0.5, 0.25)})
        run_dir = run.run_dir

    monkeypatch.setattr(parse, "_load_dataset_size", lambda _task: 1)
    monkeypatch.setattr(parse, "_evaluate_answer", _score)
    monkeypatch.setattr(parse, "_print_run_metrics", lambda *_args: None)
    monkeypatch.setattr(
        parse,
        "upload_run_metrics",
        lambda *_args, **_kwargs: pytest.fail("unexpected W&B activity"),
    )
    parse._run_directory(
        SimpleNamespace(
            run_dir=run_dir,
            log_to_wandb=False,
            wandb_project=None,
            wandb_entity=None,
        )
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["tasks"]["squad"]["dataset_size"] == 1
    assert metrics["tasks"]["squad"]["ratios"]["0.2"]["score"] == 50.0


def test_finalize_task_rebuilds_metrics_and_uploads_only_that_task(
    tmp_path, monkeypatch
):
    with _new_run(tmp_path) as run:
        run_dir = run.run_dir
        _write_example(run_dir, "first", 0, {0.2: (0.5, 0.25)})
        _write_example(run_dir, "second", 0, {0.2: (0.5, 0.25)})
        run.write_metrics(
            {
                "tasks": {"first": {"dataset_size": 1}},
                "average_relative_performance": {},
            }
        )
        task_metrics = _complete_metrics()["tasks"]["squad"]
        rebuilt = {
            "tasks": {"first": task_metrics, "second": task_metrics},
            "average_relative_performance": {},
        }
        build_calls = []
        uploads = []

        def build_metrics(given_run, *, dataset_sizes):
            build_calls.append((given_run, dataset_sizes))
            return rebuilt

        monkeypatch.setattr(parse, "build_run_metrics", build_metrics)
        monkeypatch.setattr(parse, "_print_run_metrics", lambda *_args: None)
        monkeypatch.setattr(
            parse,
            "upload_run_metrics",
            lambda metrics, *_args, **_kwargs: uploads.append(metrics) or 3,
        )

        merged = parse.finalize_task(
            run,
            "second",
            7,
            log_to_wandb=True,
            wandb_project="project",
        )

    assert build_calls == [(run, {"first": 1, "second": 7})]
    assert merged == rebuilt
    assert json.loads((run_dir / "metrics.json").read_text()) == merged
    assert list(uploads[0]["tasks"]) == ["second"]


def test_finalize_task_fails_requested_wandb_upload_for_partial_task(
    tmp_path, monkeypatch
):
    with _new_run(tmp_path) as run:
        _write_example(run.run_dir, "squad", 0, {0.2: (0.5, 0.25)})
        current = {
            "tasks": {"squad": {"complete": False, "ratios": {}}},
            "average_relative_performance": {},
        }
        monkeypatch.setattr(parse, "build_run_metrics", lambda *_args, **_kwargs: current)
        monkeypatch.setattr(parse, "_print_run_metrics", lambda *_args: None)
        with pytest.raises(ValueError, match="full squad benchmark"):
            parse.finalize_task(
                run,
                "squad",
                101,
                log_to_wandb=True,
                wandb_project="project",
            )
        assert json.loads(run.metrics_path.read_text()) == current


def test_legacy_parser_layout_remains_supported(tmp_path, monkeypatch, capsys):
    output = tmp_path / "results" / "squad" / "0_model" / "output-pair.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(
            {
                "qa": [
                    [
                        [0.2, 0.25, 0.0],
                        {"pruned": "yes", "full__": "yes", "answer": "yes"},
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parse, "_evaluate_answer", lambda *_args, **_kwargs: [1.0])

    parse._run_legacy(
        SimpleNamespace(
            level="pair",
            model="model",
            ratios=[0.2],
            tag="",
            data="squad",
            task="qa",
            num=None,
        )
    )

    text = capsys.readouterr().out
    assert "Evaluate squad on 1 samples, model" in text
    assert text.count("100.00") == 4
