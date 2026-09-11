import json
from types import SimpleNamespace

import pytest

from results.sample_store import SampleStore, build_sample_metrics
from results.evaluation_run import atomic_write_json


def sample(index, text="same answer"):
    return dict(index=index, seed=index + 42, text=text, token_count=2, finish_reason="eos")


def open_store(path, **kwargs):
    return SampleStore(path, identity={"model": "test"}, num_generations=2,
                       metadata={"length_bin": "short", "document_tokens": 50}, **kwargs)


def test_resume_preserves_duplicate_text_and_checks_identity(tmp_path):
    path = tmp_path / "samples.json"
    store = open_store(path)
    store.add_sample(1, sample(1), actual_retention=1)
    assert open_store(path).missing_indices(1) == [0]
    with pytest.raises(ValueError, match="incomplete"):
        store.texts(1)
    store.add_sample(1, sample(0), actual_retention=1)
    assert open_store(path).texts(1) == ["same answer", "same answer"]
    with pytest.raises(ValueError, match="identity"):
        SampleStore(path, identity={"model": "changed"}, num_generations=2,
                    metadata={"length_bin": "short", "document_tokens": 50})
    with pytest.raises(ValueError, match="conflict"):
        store.add_sample(1, sample(0, "changed"), actual_retention=1)
    assert len(json.loads(path.read_text())["ratios"]["1.0"]["samples"]) == 2


def test_restart_incomplete_pool_is_atomic_and_preserves_complete_pools(tmp_path, monkeypatch):
    import results.sample_store as module

    path = tmp_path / "samples.json"
    store = open_store(path)
    for i in range(2):
        store.add_sample(1, sample(i), actual_retention=1)
    store.add_sample(.5, sample(0, "superseded"), actual_retention=.51)
    previous = json.loads(path.read_text())
    original_write = module.atomic_write_json

    def fail_write(*args):
        raise OSError("write interrupted")

    monkeypatch.setattr(module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        store.restart_incomplete_pool(.5)
    assert store.data == previous == json.loads(path.read_text())
    monkeypatch.setattr(module, "atomic_write_json", original_write)
    store.restart_incomplete_pool(.5)
    restarted = open_store(path)
    pool = restarted.data["ratios"]["0.5"]
    assert pool["samples"] == [] and pool["actual_retention"] is None
    assert pool["superseded_pools"][0]["samples"] == previous["ratios"]["0.5"]["samples"]
    assert pool["superseded_pools"][0]["actual_retention"] == .51
    assert restarted.data["ratios"]["1.0"] == previous["ratios"]["1.0"]
    saved = path.read_bytes()
    restarted.restart_incomplete_pool(.5)
    restarted.restart_incomplete_pool(1)
    restarted.restart_incomplete_pool(.25)
    assert path.read_bytes() == saved
    restarted.add_sample(.5, sample(0, "fresh"), actual_retention=.49)
    with pytest.raises(ValueError, match="actual retention conflict"):
        restarted.add_sample(.5, sample(1), actual_retention=.51)


def test_superseded_pool_does_not_contribute_to_metrics(tmp_path):
    folder = tmp_path / "samples/summary_test"
    atomic_write_json(folder / "manifest.json", {
        "requested_ratios": [1, .5], "selected_indices": [0], "dataset_size": 1,
    })
    store = open_store(folder / "examples/0.json")
    for i in range(2):
        store.add_sample(1, sample(i), actual_retention=1)
    store.add_sample(.5, sample(0), actual_retention=.51)
    store.restart_incomplete_pool(.5)
    for i in range(2):
        store.add_sample(.5, sample(i, "unrelated words"), actual_retention=.49)
    metric = build_sample_metrics(SimpleNamespace(run_dir=tmp_path))["summary_test"]
    assert metric["complete"]
    assert metric["ratios"]["0.5"]["rougeL"] == 0
    assert metric["ratios"]["0.5"]["actual_retention"] == .49


def test_common_cohort_withholds_scores_until_all_ratios_complete(tmp_path):
    run = SimpleNamespace(run_dir=tmp_path)
    folder = tmp_path / "samples" / "summary_test"
    atomic_write_json(folder / "manifest.json", {
        "requested_ratios": [1, .5, .5], "selected_indices": [0, 2],
        "dataset_size": 3, "excluded": [{"index": 1, "reason": "context_limit"}],
    })
    for index in [0, 2]:
        store = open_store(folder / "examples" / f"{index}.json")
        for ratio in [1, .5]:
            for j in range(2 if index == 0 or ratio == 1 else 1):
                store.add_sample(ratio, sample(j), actual_retention=ratio)
    metrics = build_sample_metrics(run)["summary_test"]
    assert not metrics["complete"]
    assert metrics["example_count"] == 1
    assert metrics["selected_count"] == 2
    assert all(value["score"] is None for value in metrics["ratios"].values())
    assert metrics["ratios"]["0.5"]["example_count"] == 1
    open_store(folder / "examples" / "2.json").add_sample(.5, sample(1), actual_retention=.5)
    metrics = build_sample_metrics(run)["summary_test"]
    assert metrics["cohort_complete"]
    assert not metrics["complete"]
    assert metrics["ratios"]["0.5"]["score"] == 100
    assert metrics["length_bins"]["short"]["complete"]
    assert "relative" not in metrics["ratios"]["0.5"]


def test_summary_only_parse_and_missing_score_logging(tmp_path, capsys):
    from results.parse import build_run_metrics, _dataset_sizes, _print_run_metrics, _wandb_points

    folder = tmp_path / "samples" / "summary_test"
    atomic_write_json(folder / "manifest.json", {
        "requested_ratios": [1, .5], "selected_indices": [0], "dataset_size": 3,
        "inventory": [{"index": 0, "length_bin": "short"}], "num_generations": 2,
    })
    run = SimpleNamespace(run_dir=tmp_path, outputs_dir=tmp_path / "outputs",
                          dataset_sizes={}, manifest={}, iter_examples=lambda: iter(()))
    assert _dataset_sizes(run, {}) == {"summary_test": 3}
    metrics = build_run_metrics(run, dataset_sizes={"summary_test": 3})
    assert metrics["average_relative_performance"] == {}
    assert metrics["tasks"]["summary_test"]["length_bins"]["short"]["selected_count"] == 1
    assert _wandb_points(metrics) == {}
    _print_run_metrics(tmp_path, metrics, "pair")
    assert "incomplete" in capsys.readouterr().out


def test_pilot_scores_do_not_authorize_full_benchmark_logging(tmp_path):
    from results.parse import _require_full_benchmarks, _task_is_complete

    folder = tmp_path / "samples" / "summary_test"
    manifest = {"requested_ratios": [1, .5], "selected_indices": [0],
                "dataset_size": 100, "source_inventory": {"total": 150}}
    atomic_write_json(folder / "manifest.json", manifest)
    store = open_store(folder / "examples" / "0.json")
    for ratio in [1, .5]:
        for index in range(2):
            store.add_sample(ratio, sample(index), actual_retention=ratio)
    run = SimpleNamespace(run_dir=tmp_path)
    metrics = build_sample_metrics(run)
    task = metrics["summary_test"]
    assert task["cohort_complete"] and not task["complete"]
    assert task["ratios"]["0.5"]["score"] == 100
    assert not task["ratios"]["0.5"]["complete"]
    assert task["source_inventory"] == {"total": 150}
    assert not _task_is_complete(task)
    with pytest.raises(ValueError, match="full"):
        _require_full_benchmarks({"tasks": metrics})
    manifest["dataset_size"] = 1
    atomic_write_json(folder / "manifest.json", manifest)
    task = build_sample_metrics(run)["summary_test"]
    assert task["complete"] and _task_is_complete(task)
    _require_full_benchmarks({"tasks": {"summary_test": task}})


def test_parser_checks_manifest_identity_and_example_index(tmp_path):
    import hashlib

    folder = tmp_path / "samples" / "summary_test"
    identity = {"model": "test"}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    atomic_write_json(folder / "manifest.json", {
        "requested_ratios": [1], "selected_indices": [0], "dataset_size": 1,
        "example_identities": {"0": identity_hash},
    })
    path = folder / "examples" / "0.json"
    store = open_store(path)
    store.add_sample(1, sample(0))
    run = SimpleNamespace(run_dir=tmp_path)
    assert not build_sample_metrics(run)["summary_test"]["complete"]
    changed = json.loads(path.read_text())
    changed["identity"]["model"] = "wrong"
    atomic_write_json(path, changed)
    with pytest.raises(ValueError, match="identity"):
        build_sample_metrics(run)
    changed["identity"] = identity
    changed["metadata"]["index"] = 1
    atomic_write_json(path, changed)
    with pytest.raises(ValueError, match="index"):
        build_sample_metrics(run)


def test_length_bins_count_eligible_inventory_and_keep_unselected_bins(tmp_path):
    folder = tmp_path / "samples" / "summary_test"
    atomic_write_json(folder / "manifest.json", {
        "requested_ratios": [1], "selected_indices": [0], "dataset_size": 3,
        "inventory": [{"index": 0, "length_bin": "short"},
                      {"index": 1, "length_bin": "short"},
                      {"index": 2, "length_bin": "medium"},
                      {"index": 3, "length_bin": "long"}],
        "excluded": [{"index": 3, "reason": "context_limit"}],
        "source_inventory": {"length_bins": {"short": 2, "medium": 1, "long": 1, "huge": 0}},
    })
    store = open_store(folder / "examples" / "0.json")
    for index in range(2):
        store.add_sample(1, sample(index), actual_retention=1)
    bins = build_sample_metrics(SimpleNamespace(run_dir=tmp_path))["summary_test"]["length_bins"]
    assert set(bins) == {"short", "medium", "long", "huge"}
    assert bins["short"]["dataset_size"] == 2
    assert bins["short"]["cohort_complete"] and not bins["short"]["complete"]
    assert bins["short"]["ratios"]["1.0"]["score"] == 100
    assert bins["medium"]["dataset_size"] == 1
    assert bins["long"]["dataset_size"] == bins["huge"]["dataset_size"] == 0
    for label in ["medium", "long", "huge"]:
        assert bins[label]["selected_count"] == 0
        assert not bins[label]["complete"]
        assert bins[label]["ratios"]["1.0"]["score"] is None


def test_optional_null_source_inventory(tmp_path):
    atomic_write_json(tmp_path / "samples" / "summary_test" / "manifest.json", {
        "requested_ratios": [1], "selected_indices": [0], "dataset_size": 1,
        "source_inventory": None,
    })
    metrics = build_sample_metrics(SimpleNamespace(run_dir=tmp_path))["summary_test"]
    assert not metrics["complete"]
    assert metrics["length_bins"]["unknown"]["selected_count"] == 1
