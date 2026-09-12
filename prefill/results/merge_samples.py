"""Collect independent summary shards and recompute metrics from their samples.

Run from prefill: python -m results.merge_samples --output MERGED SHARD [SHARD ...]
The destination must be new. Use --allow-partial for an incomplete snapshot.
"""

import argparse
import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path

from results.evaluation_run import EvaluationRun, atomic_write_json
from results.sample_store import SampleStore, _ratio_key, sample_manifests


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _semantic_protocol(protocol):
    protocol = copy.deepcopy(protocol)
    pruning = protocol["pruning"]
    if pruning.get("checkpoint_path"):
        if not re.fullmatch(r"[0-9a-f]{64}", pruning.get("checkpoint_sha256", "")):
            raise ValueError("invalid checkpoint fingerprint in summary protocol")
        # Account-local copies have different paths but identical model weights.
        # Keep the original path inside each sample identity (and its RNG seed).
        del pruning["checkpoint_path"]
    return protocol


def _signature(manifest):
    value = {key: item for key, item in manifest.items()
             if key not in {"selected_indices", "example_identities"}}
    value["protocol"] = _semantic_protocol(value["protocol"])
    value["requested_ratios"] = sorted({_ratio_key(r) for r in [1, *value["requested_ratios"]]})
    return value


def _validate_task(run, task, manifest):
    selected = manifest["selected_indices"]
    inventory = manifest["inventory"]
    indices = [item["index"] for item in inventory]
    excluded = {item["index"] for item in manifest["excluded"]}
    eligible = set(indices) - excluded
    count = manifest["num_generations"]
    if (not all(type(i) is int and i >= 0 for i in [*indices, *selected])
            or len(set(indices)) != len(indices) or len(set(selected)) != len(selected)
            or not excluded <= set(indices) or not set(selected) <= eligible
            or manifest["dataset_size"] != len(eligible)
            or type(count) is not int or count < 1
            or set(manifest["example_identities"]) != {str(i) for i in selected}):
        raise ValueError(f"invalid summary cohort: {run.run_dir}/{task}")
    protocol = manifest["protocol"]
    if (protocol["generation"]["num_generations"] != count
            or protocol["generation"] != run.manifest.get("generation_settings")
            or any(protocol["pruning"].get(key) != value for key, value in run.manifest.items())
            or run.dataset_sizes.get(task, len(eligible)) != len(eligible)):
        raise ValueError(f"summary protocol mismatch with run manifest: {run.run_dir}/{task}")
    return {item["index"]: item for item in inventory}


def merge_sample_runs(shard_dirs, output, *, allow_partial=False):
    """Validate and combine raw pools, preserving original identities and seeds.

    Shards must describe the same experiment, tasks, and dataset inventory. Disjoint
    documents may use byte-identical checkpoints at different account paths.
    Overlapping documents must have identical identities and matching samples;
    complementary sample indices from interrupted copies can be combined.
    Previous merged snapshots can be inputs to a new collection.
    """
    from results.parse import build_run_metrics

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"merge destination already exists: {output}")
    sources = list(dict.fromkeys(Path(path).resolve() for path in shard_dirs))
    if not sources:
        raise ValueError("at least one summary shard is required")
    runs = [EvaluationRun.load(path) for path in sources]
    tasks, signatures, provenance = {}, {}, []
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build privately so invalid or incomplete collections never publish a run.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        atomic_write_json(staging / "manifest.json", runs[0].manifest)
        for run in runs:
            manifests = sample_manifests(run)
            if not manifests:
                raise ValueError(f"run has no summarization shards: {run.run_dir}")
            if tasks and manifests.keys() != tasks.keys():
                raise ValueError("summary shard task set mismatch; merge each method/dataset group separately")
            if any(run.iter_examples()):
                raise ValueError(f"merge accepts summary-only runs: {run.run_dir}")
            origin = {"run_dir": str(run.run_dir), "manifest": run.manifest, "tasks": manifests}
            previous_shards = run.run_dir / "shards.json"
            if previous_shards.exists():
                origin["shards"] = json.loads(previous_shards.read_text(encoding="utf-8"))
            provenance.append(origin)
            for task, manifest in manifests.items():
                inventory = _validate_task(run, task, manifest)
                signature = _signature(manifest)
                if task in signatures and signatures[task] != signature:
                    raise ValueError(f"summary shard protocol or inventory mismatch: {task}")
                if task not in tasks:
                    signatures[task] = signature
                    tasks[task] = {**copy.deepcopy(manifest), "selected_indices": [],
                                   "example_identities": {}}
                merged = tasks[task]
                selected = manifest["selected_indices"]
                merged["selected_indices"] = sorted(set(merged["selected_indices"]) | set(selected))
                for index in selected:
                    identity = manifest["example_identities"][str(index)]
                    if merged["example_identities"].get(str(index), identity) != identity:
                        raise ValueError(f"overlapping sample identity conflict: {task}/{index}")
                    merged["example_identities"][str(index)] = identity
                folder = run.run_dir / "samples" / task / "examples"
                declared = {f"{index}.json" for index in selected}
                for path in sorted(folder.glob("*.json")):
                    if path.name not in declared:
                        raise ValueError(f"undeclared sample file: {path}")
                    data = json.loads(path.read_text(encoding="utf-8"))
                    index = int(path.stem)
                    if (_digest(data["identity"]) != manifest["example_identities"][str(index)]
                            or _semantic_protocol({key: data["identity"].get(key)
                                for key in ("runtime", "generation", "pruning")}) != signature["protocol"]):
                        raise ValueError(f"sample identity differs from shard manifest: {path}")
                    if data["metadata"] != inventory[index] or data["num_generations"] != manifest["num_generations"]:
                        raise ValueError(f"sample metadata or generation count mismatch: {path}")
                    source = SampleStore(path, identity=data["identity"],
                                         num_generations=data["num_generations"], metadata=data["metadata"])
                    if not set(source.ratios) <= {float(r) for r in signature["requested_ratios"]}:
                        raise ValueError(f"sample contains an undeclared ratio: {path}")
                    destination = SampleStore(staging / "samples" / task / "examples" / path.name,
                        identity=data["identity"], num_generations=data["num_generations"], metadata=data["metadata"])
                    for ratio, condition in source.data["ratios"].items():
                        for sample in condition["samples"]:
                            destination.add_sample(ratio, sample, actual_retention=condition["actual_retention"])

        for task, manifest in tasks.items():
            atomic_write_json(staging / "samples" / task / "manifest.json", manifest)
        atomic_write_json(staging / "datasets.json", {task: m["dataset_size"] for task, m in tasks.items()})
        atomic_write_json(staging / "shards.json", provenance)
        merged_run = EvaluationRun.load(staging)
        metrics = build_run_metrics(merged_run, dataset_sizes=merged_run.dataset_sizes)
        incomplete = [task for task, result in metrics["tasks"].items() if not result["complete"]]
        if incomplete and not allow_partial:
            raise ValueError(f"incomplete summary datasets: {', '.join(incomplete)}; use --allow-partial for a snapshot")
        merged_run.write_metrics(metrics)
        staging.rename(output)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new combined run directory")
    parser.add_argument("--allow-partial", action="store_true", help="allow incomplete datasets or sample pools")
    parser.add_argument("shards", nargs="+", type=Path, help="evaluation run directories to collect")
    args = parser.parse_args(argv)
    metrics = merge_sample_runs(args.shards, args.output, allow_partial=args.allow_partial)
    for task, result in metrics["tasks"].items():
        status = "complete" if result["complete"] else "incomplete"
        print(f"{task}: {result['example_count']}/{result['dataset_size']} documents, {status}")
    print(f"Merged samples and metrics: {args.output.resolve()}")


if __name__ == "__main__":
    main()
