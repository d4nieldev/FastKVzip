"""Upload completed evaluation workers in one serial, read-only collection pass."""

import json
import math
from pathlib import Path

from data.benchmarks import get_data_list, parse_ruler_name
from data.ruler import RULER_REVISIONS, RULER_SAMPLES
from generation import GENERATION_REVISION
from results.evaluation_run import EvaluationRun
from results.parse import ruler_macro_averages, upload_run_metrics
from window import WINDOW_REVISION


def _complete(task):
    size = task.get("dataset_size", 0)
    ratios = task.get("ratios", {})
    full = task.get("full_cache", {})
    return (
        task.get("complete") is True
        and size > 0
        and task.get("example_count") == size
        and full.get("complete") is True
        and full.get("example_count") == size
        and set(ratios) == {"1.0", "0.75", "0.5", "0.4", "0.3", "0.2"}
        and all(
            values.get("complete") is True
            and values.get("example_count") == values.get("dataset_size") == size
            for values in ratios.values()
        )
    )


def upload_completed_runs(
    run_dirs, *, wandb_run_id, project, entity, wandb_module=None
):
    """Validate every snapshot before serial uploads; never acquire worker locks."""
    expected = get_data_list("all")
    protocol = {
        "wandb_run_id": wandb_run_id,
        "window_size": 0,
        "window_revision": WINDOW_REVISION,
        "level": "pair",
        "prefill_mode": "post-prefill",
        "ruler_prompt_mode": "graphkv",
        "generation_revision": GENERATION_REVISION,
    }
    tasks = {}
    manifest = None
    for path in run_dirs:
        path = Path(path)
        if not (path / "manifest.json").exists():
            continue
        run = EvaluationRun.load(path)
        manifest = run.manifest
        for field, value in protocol.items():
            if manifest[field] != value:
                raise ValueError(f"{path}: incompatible {field}")
        if run.metrics_path.exists():
            sizes = run.dataset_sizes
            for name, values in json.loads(run.metrics_path.read_text())[
                "tasks"
            ].items():
                if name not in expected:
                    raise ValueError(f"{path}: unapproved production task {name}")
                if name not in sizes or values.get("dataset_size") != sizes[name]:
                    raise ValueError(
                        f"{path}: {name} metrics do not match the recorded dataset size"
                    )
                if name.startswith("ruler_"):
                    _, length = parse_ruler_name(name)
                    if (
                        manifest["dataset_revisions"].get(f"ruler_{length}")
                        != RULER_REVISIONS[length]
                    ):
                        raise ValueError(
                            f"{path}: {name} has an incompatible dataset revision"
                        )
                    if sizes[name] != RULER_SAMPLES:
                        raise ValueError(
                            f"{path}: {name} dataset size must be {RULER_SAMPLES}"
                        )
                if _complete(values):
                    for point in [values["full_cache"], *values["ratios"].values()]:
                        for field in (
                            "score",
                            "relative",
                            "actual_retention",
                            "model_selection_rate",
                        ):
                            value = point.get(field)
                            if (field == "score" or field in point) and (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                            ):
                                raise ValueError(
                                    f"{path}: invalid {field} metric for {name}"
                                )
                    if name in tasks and tasks[name] != values:
                        raise ValueError(
                            f"{name}: conflicting completed worker metrics"
                        )
                    tasks[name] = values
    uploaded = (
        upload_run_metrics(
            {"tasks": tasks},
            manifest,
            project=project,
            entity=entity,
            wandb_module=wandb_module,
        )
        if tasks
        else 0
    )
    return {
        "tasks": tasks,
        "uploaded_points": uploaded,
        "coverage": {
            "completed": len(tasks),
            "expected": len(expected),
            "missing": [name for name in expected if name not in tasks],
        },
        "ruler_macro_averages": ruler_macro_averages(tasks),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    args = parser.parse_args()
    try:
        summary = upload_completed_runs(
            args.run_dirs,
            wandb_run_id=args.wandb_run_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
