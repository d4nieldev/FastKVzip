"""Atomic sample checkpoints and matched-distribution summary metrics."""

import copy
import hashlib
import json
import math
from pathlib import Path

from results.evaluation_run import atomic_write_json


def _ratio_key(ratio):
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("sample retention ratio must be in (0, 1]")
    return str(ratio)


class SampleStore:
    def __init__(self, path, *, identity, num_generations, metadata):
        self.path = Path(path)
        if type(num_generations) is not int or num_generations < 1:
            raise ValueError("num_generations must be a positive integer")
        # JSON normalization allows tuple-valued identities to survive a round trip.
        expected = json.loads(json.dumps(dict(
            version=1, identity=identity, num_generations=num_generations,
            metadata=metadata,
        ), allow_nan=False))
        self.data = {**expected, "ratios": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if any(self.data.get(key) != value for key, value in expected.items()):
                raise ValueError(f"sample identity or metadata mismatch: {self.path}")
            self._validate()

    def _validate_sample(self, sample):
        index = sample.get("index")
        if type(index) is not int or not 0 <= index < self.data["num_generations"]:
            raise ValueError("sample index outside configured generation count")
        if (type(sample.get("seed")) is not int
                or not isinstance(sample.get("text"), str)
                or type(sample.get("token_count")) is not int
                or sample["token_count"] < 0
                or not isinstance(sample.get("finish_reason"), str)):
            raise ValueError("invalid generated sample")

    def _validate(self):
        for key, condition in self.data["ratios"].items():
            if _ratio_key(key) != key:
                raise ValueError("noncanonical sample ratio")
            seen = set()
            for sample in condition["samples"]:
                self._validate_sample(sample)
                if sample["index"] in seen:
                    raise ValueError("duplicate sample index")
                seen.add(sample["index"])
            retention = condition.get("actual_retention")
            if retention is not None and (not math.isfinite(retention) or not 0 <= retention <= 1):
                raise ValueError("invalid actual retention")

    @property
    def ratios(self):
        return [float(key) for key in self.data["ratios"]]

    def missing_indices(self, ratio):
        saved = {sample["index"] for sample in self.data["ratios"].get(
            _ratio_key(ratio), {"samples": []})["samples"]}
        return [index for index in range(self.data["num_generations"]) if index not in saved]

    def is_complete(self, ratio):
        return not self.missing_indices(ratio)

    def texts(self, ratio):
        if not self.is_complete(ratio):
            raise ValueError(f"incomplete sample condition: {ratio}")
        return [sample["text"] for sample in sorted(
            self.data["ratios"][_ratio_key(ratio)]["samples"], key=lambda s: s["index"])]

    def add_sample(self, ratio, sample, *, actual_retention=None):
        self._validate_sample(sample)
        key = _ratio_key(ratio)
        updated = copy.deepcopy(self.data)
        condition = updated["ratios"].setdefault(key, {"samples": [], "actual_retention": None})
        if actual_retention is not None:
            actual_retention = float(actual_retention)
            if not math.isfinite(actual_retention) or not 0 <= actual_retention <= 1:
                raise ValueError("invalid actual retention")
            if condition["actual_retention"] not in (None, actual_retention):
                raise ValueError("actual retention conflict")
            condition["actual_retention"] = actual_retention
        previous = next((s for s in condition["samples"] if s["index"] == sample["index"]), None)
        normalized = json.loads(json.dumps(sample, allow_nan=False))
        if previous is not None and previous != normalized:
            raise ValueError("saved sample conflict")
        if previous is None:
            condition["samples"].append(normalized)
            condition["samples"].sort(key=lambda s: s["index"])
        atomic_write_json(self.path, updated)
        self.data = updated


def sample_manifests(run):
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((Path(run.run_dir) / "samples").glob("*/manifest.json"))}


def build_sample_metrics(run):
    from results.summary_metric import matched_rouge

    tasks = {}
    for task, manifest in sample_manifests(run).items():
        ratios = list(dict.fromkeys(["1.0", *(_ratio_key(r) for r in manifest["requested_ratios"])]))
        selected = manifest["selected_indices"]
        if len(set(selected)) != len(selected):
            raise ValueError(f"duplicate selected sample index: {task}")
        stores = {}
        for index in selected:
            path = Path(run.run_dir) / "samples" / task / "examples" / f"{index}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if "example_identities" in manifest:
                    digest = hashlib.sha256(json.dumps(data["identity"], sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                    if manifest["example_identities"].get(str(index)) != digest:
                        raise ValueError(f"sample identity differs from task manifest: {task}/{index}")
                if data["metadata"].get("index", index) != index:
                    raise ValueError(f"sample metadata index differs from filename: {task}/{index}")
                stores[index] = SampleStore(path, identity=data["identity"],
                    num_generations=data["num_generations"], metadata=data["metadata"])
        if len({s.data["num_generations"] for s in stores.values()}) > 1:
            raise ValueError(f"mixed generation counts: {task}")
        if "num_generations" in manifest and any(
                s.data["num_generations"] != manifest["num_generations"] for s in stores.values()):
            raise ValueError(f"generation count differs from task manifest: {task}")
        common = {index for index, store in stores.items()
                  if all(store.is_complete(ratio) for ratio in ratios)}
        scores = {index: {ratio: matched_rouge(stores[index].texts(1), stores[index].texts(ratio))
                          for ratio in ratios} for index in common}

        def aggregate(indices, dataset_size):
            cohort = set(indices) & common
            cohort_complete = bool(indices) and len(cohort) == len(indices)
            complete = cohort_complete and len(indices) == dataset_size
            result = dict(metric="matched_rouge", example_count=len(cohort),
                          dataset_size=dataset_size, selected_count=len(indices),
                          complete=complete, cohort_complete=cohort_complete, ratios={})
            for ratio in ratios:
                available = [stores[i] for i in indices if i in stores and stores[i].is_complete(ratio)]
                retention = [s.data["ratios"][ratio]["actual_retention"] for s in available
                             if s.data["ratios"][ratio]["actual_retention"] is not None]
                values = {metric: sum(scores[i][ratio][metric] for i in cohort) / len(cohort) * 100
                          if cohort_complete else None for metric in ("rouge1", "rouge2", "rougeL")}
                result["ratios"][ratio] = dict(**values, score=values["rougeL"],
                    actual_retention=sum(retention) / len(retention) if retention else None,
                    example_count=len(cohort), condition_complete_count=len(available),
                    dataset_size=dataset_size, selected_count=len(indices), complete=complete,
                    cohort_complete=cohort_complete)
            result["full_cache"] = {key: result["ratios"]["1.0"][key]
                                    for key in ("score", "example_count", "complete", "cohort_complete")}
            return result

        result = aggregate(selected, manifest["dataset_size"])
        result["excluded"] = manifest.get("excluded", [])
        result["inventory"] = manifest.get("inventory", {})
        result["source_inventory"] = manifest.get("source_inventory", {})
        # Full inventory includes unselected and model-limit-excluded documents.
        bins = {}
        inventory = manifest.get("inventory", [])
        if isinstance(inventory, dict):
            inventory = list(inventory.values())
        for item in inventory:
            if isinstance(item, dict) and "index" in item and item.get("length_bin") is not None:
                bins[item["index"]] = str(item["length_bin"])
        for index, store in stores.items():
            bins[index] = str(store.data["metadata"].get("length_bin", "unknown"))
        excluded = {item["index"] for item in manifest.get("excluded", []) if "index" in item}
        labels = set(bins.values()) | {bins.get(i, "unknown") for i in selected}
        labels.update((manifest.get("source_inventory") or {}).get("length_bins", {}))
        result["length_bins"] = {label: aggregate([i for i in selected if bins.get(i, "unknown") == label],
                                                sum(bins.get(i, "unknown") == label
                                                    for i in (set(bins) | set(selected)) - excluded))
                                 for label in sorted(labels)}
        tasks[task] = result
    return tasks
