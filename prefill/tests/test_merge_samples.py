import copy
import hashlib
import json

import pytest

from results.evaluation_run import EvaluationRun, atomic_write_json
from results.sample_store import SampleStore


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def shard(tmp_path, name, selected, *, checkpoint=None, missing=None):
    checkpoint = checkpoint or tmp_path / name / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"identical checkpoint weights")
    settings = dict(temperature=.7, top_p=.9, top_k=0, max_new_tokens=1024, num_generations=2)
    run = EvaluationRun.open(tmp_path / "runs", name, checkpoint_path=checkpoint,
        wandb_run_id="training", window_size=0, level="pair", generation_settings=settings)
    protocol = {"runtime": {"model": "test", "model_revision": "a" * 40},
                "generation": settings,
                "pruning": {**run.manifest, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    inventory = [{"index": i, "document_id": f"doc-{i}", "length_bin": "short" if i < 2 else "long",
                  "document_tokens": 10000 + 10000 * i, "prompt_tokens": 10100 + 10000 * i}
                 for i in range(4)]
    manifest = {"protocol": protocol, "num_generations": 2, "requested_ratios": [1, .5],
                "selected_indices": selected, "dataset_size": 3,
                "excluded": [{**inventory[3], "reason": "context_limit"}],
                "inventory": inventory, "source_inventory": {"source_size": 4},
                "context_limit": 40000, "example_identities": {}}
    folder = run.run_dir / "samples" / "govreport_summary"
    for i in selected:
        identity = {**copy.deepcopy(protocol), "input_sha256": f"prompt-{i}",
                    "document_sha256": f"document-{i}", "source": {"index": i}}
        manifest["example_identities"][str(i)] = digest(identity)
        store = SampleStore(folder / "examples" / f"{i}.json", identity=identity,
                            num_generations=2, metadata=inventory[i])
        for ratio in [1, .5]:
            for sample_index in range(2):
                if (i, ratio, sample_index) == missing:
                    continue
                store.add_sample(ratio, dict(index=sample_index, seed=i * 100 + sample_index,
                    text="gamma delta" if i == 2 and ratio == .5 else "alpha beta",
                    token_count=2, finish_reason="eos"), actual_retention=ratio)
    atomic_write_json(folder / "manifest.json", manifest)
    run.record_dataset_size("govreport_summary", 3)
    return run.run_dir


def alter_manifest(run_dir, change):
    path = run_dir / "samples" / "govreport_summary" / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    atomic_write_json(path, manifest)


def test_merge_preserves_account_provenance_and_weights_documents_equally(tmp_path):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "account-one", [0, 1])
    second = shard(tmp_path, "account-two", [2])
    output = tmp_path / "merged"
    metrics = merge_sample_runs([first, second], output)
    task = metrics["tasks"]["govreport_summary"]
    assert task["complete"] and task["example_count"] == 3
    assert task["ratios"]["0.5"]["score"] == pytest.approx(200 / 3)
    assert task["length_bins"]["short"]["ratios"]["0.5"]["score"] == 100
    assert task["length_bins"]["long"]["complete"]
    assert task["length_bins"]["long"]["dataset_size"] == 1
    assert task["length_bins"]["long"]["ratios"]["0.5"]["score"] == 0
    for source, indices in [(first, [0, 1]), (second, [2])]:
        for i in indices:
            relative = f"samples/govreport_summary/examples/{i}.json"
            assert json.loads((output / relative).read_text()) == json.loads((source / relative).read_text())
    provenance = json.loads((output / "shards.json").read_text())
    assert {item["run_dir"] for item in provenance} == {str(first), str(second)}
    assert len({item["manifest"]["checkpoint_path"] for item in provenance}) == 2
    assert EvaluationRun.load(output).dataset_sizes == {"govreport_summary": 3}


def test_partial_merge_is_explicit_and_never_claims_complete_benchmark(tmp_path):
    from results.merge_samples import merge_sample_runs

    source = shard(tmp_path, "partial", [0, 2], missing=(2, .5, 1))
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="incomplete"):
        merge_sample_runs([source], output)
    assert not output.exists()
    metrics = merge_sample_runs([source], output, allow_partial=True)
    task = metrics["tasks"]["govreport_summary"]
    assert not task["complete"] and not task["cohort_complete"]
    assert task["example_count"] == 1 and task["selected_count"] == 2
    assert task["ratios"]["0.5"]["score"] is None
    assert not task["length_bins"]["short"]["complete"]
    assert task["length_bins"]["short"]["dataset_size"] == 2


def test_identical_overlap_is_deduplicated_but_sample_conflicts_fail(tmp_path):
    from results.merge_samples import merge_sample_runs

    checkpoint = tmp_path / "shared.pt"
    first = shard(tmp_path, "first", [0, 1], checkpoint=checkpoint)
    second = shard(tmp_path, "second", [1, 2], checkpoint=checkpoint)
    metrics = merge_sample_runs([first, second], tmp_path / "merged")
    assert metrics["tasks"]["govreport_summary"]["example_count"] == 3
    path = second / "samples/govreport_summary/examples/1.json"
    changed = json.loads(path.read_text())
    changed["ratios"]["0.5"]["samples"][0]["text"] = "changed"
    atomic_write_json(path, changed)
    with pytest.raises(ValueError, match="conflict"):
        merge_sample_runs([first, second], tmp_path / "conflicted")
    assert not (tmp_path / "conflicted").exists()


@pytest.mark.parametrize("field,value", [
    ("requested_ratios", [1, .25]), ("num_generations", 3),
    ("inventory", []), ("dataset_size", 4), ("context_limit", 50000),
])
def test_merge_rejects_different_cohorts_and_settings(tmp_path, field, value):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "first", [0, 1])
    second = shard(tmp_path, "second", [2])
    alter_manifest(second, lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match="mismatch|invalid"):
        merge_sample_runs([first, second], tmp_path / "merged")


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "different"), ("level", "pair-head")])
def test_merge_rejects_different_pruning(tmp_path, field, value):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "first", [0, 1])
    second = shard(tmp_path, "second", [2])
    alter_manifest(second, lambda data: data["protocol"]["pruning"].update({field: value}))
    with pytest.raises(ValueError, match="mismatch|invalid"):
        merge_sample_runs([first, second], tmp_path / "merged")


def test_merge_rejects_undeclared_or_tampered_samples(tmp_path):
    from results.merge_samples import merge_sample_runs

    source = shard(tmp_path, "source", [0, 1, 2])
    path = source / "samples/govreport_summary/examples/0.json"
    changed = json.loads(path.read_text())
    changed["identity"]["input_sha256"] = "different"
    atomic_write_json(path, changed)
    with pytest.raises(ValueError, match="identity"):
        merge_sample_runs([source], tmp_path / "merged")


def test_merger_cli_and_existing_destination(tmp_path, capsys):
    from results.merge_samples import main, merge_sample_runs

    source = shard(tmp_path, "source", [0, 1, 2])
    output = tmp_path / "merged"
    main(["--output", str(output), str(source)])
    assert "3/3" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        merge_sample_runs([source], output)


def test_different_task_groups_cannot_disguise_incompatible_runs(tmp_path):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "first", [0, 1, 2])
    second = shard(tmp_path, "second", [0, 1, 2])
    samples = second / "samples"
    (samples / "govreport_summary").rename(samples / "pg19_summary")
    with pytest.raises(ValueError, match="task.*mismatch"):
        merge_sample_runs([first, second], tmp_path / "merged")


def test_complementary_resumed_samples_are_combined_without_losing_duplicates(tmp_path):
    from results.merge_samples import merge_sample_runs

    checkpoint = tmp_path / "shared.pt"
    first = shard(tmp_path, "first", [0, 1, 2], checkpoint=checkpoint, missing=(2, .5, 1))
    second = shard(tmp_path, "second", [2], checkpoint=checkpoint, missing=(2, .5, 0))
    output = tmp_path / "merged"
    assert merge_sample_runs([first, second], output)["tasks"]["govreport_summary"]["complete"]
    path = output / "samples/govreport_summary/examples/2.json"
    data = json.loads(path.read_text())
    assert [sample["text"] for sample in data["ratios"]["0.5"]["samples"]] == ["gamma delta"] * 2


def test_cross_account_merged_output_can_be_merged_again(tmp_path):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "account-one", [0, 1])
    second = shard(tmp_path, "account-two", [2])
    merged = tmp_path / "merged"
    original_metrics = merge_sample_runs([first, second], merged)
    repeated = tmp_path / "repeated"
    assert merge_sample_runs([merged], repeated) == original_metrics
    for index in range(3):
        relative = f"samples/govreport_summary/examples/{index}.json"
        assert json.loads((repeated / relative).read_text()) == json.loads((merged / relative).read_text())
    provenance = json.loads((repeated / "shards.json").read_text())
    assert provenance[0]["shards"] == json.loads((merged / "shards.json").read_text())


def test_cross_account_partial_merge_can_be_extended(tmp_path):
    from results.merge_samples import merge_sample_runs

    first = shard(tmp_path, "account-one", [0])
    second = shard(tmp_path, "account-two", [1])
    final = shard(tmp_path, "account-three", [2])
    partial = tmp_path / "partial"
    assert not merge_sample_runs([first, second], partial, allow_partial=True)["tasks"]["govreport_summary"]["complete"]
    complete = tmp_path / "complete"
    metrics = merge_sample_runs([partial, final], complete)
    assert metrics["tasks"]["govreport_summary"]["complete"]
    assert metrics["tasks"]["govreport_summary"]["ratios"]["0.5"]["score"] == pytest.approx(200 / 3)


def test_semantic_sample_protocol_must_match_even_with_matching_identity_digest(tmp_path):
    from results.merge_samples import merge_sample_runs

    source = shard(tmp_path, "source", [0, 1, 2])
    path = source / "samples/govreport_summary/examples/0.json"
    data = json.loads(path.read_text())
    data["identity"]["runtime"]["model_revision"] = "b" * 40
    atomic_write_json(path, data)
    alter_manifest(source, lambda manifest: manifest["example_identities"].update({"0": digest(data["identity"])}))
    with pytest.raises(ValueError, match="identity"):
        merge_sample_runs([source], tmp_path / "merged")
