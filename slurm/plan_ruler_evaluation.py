"""Build the approved evaluation grid offline; never contact or submit to Slurm.

Read the specification as JSON from stdin, emit the exact manifest for approval.
The test fixture documents its small input format. Resource profiles and per-task
runtime estimates must cite measured evidence; account observations come from
fresh preflight checks, not this offline planner. Recheck those observations
before executing any approved command.

An archived ``completed_checkpoint_approval`` may waive an expired scheduler
dependency; retain its ``job_id``, ``approved_at`` and ``reference`` in the spec.

The caller owns durable receipt writes: append a ``submitting`` attempt BEFORE
calling sbatch, then record its returned job ID or mark the outcome ``unknown``.
Do not retry unknown outcomes. Reconcile them while retaining attempt history.
"""

import copy
import hashlib
import json
import math
import re
import runpy
from pathlib import Path


def build_manifest(spec):
    """Order all 107 benchmarks and balance estimated GPU work over live slots."""
    inventory = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "prefill/data/benchmarks.py")
    )["get_data_list"]("all")
    benchmarks = {row["data"]: row for row in spec["benchmarks"]}
    if len(benchmarks) != len(spec["benchmarks"]) or set(benchmarks) != set(inventory):
        raise ValueError(
            "benchmark inventory must contain exactly the approved 107 datasets"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec["name"]):
        raise ValueError("invalid experiment name")
    if any(not spec["wandb"].get(field) for field in ("entity", "project", "run_id")):
        raise ValueError("one explicit new evaluation W&B destination is required")
    if (
        not spec.get("source_wandb_run_id")
        or spec["wandb"]["run_id"] == spec["source_wandb_run_id"]
    ):
        raise ValueError(
            "the new evaluation destination must differ from the source training run"
        )
    for benchmark in benchmarks.values():
        estimate = benchmark.get("estimated_seconds")
        if (
            not isinstance(estimate, (float, int))
            or not math.isfinite(estimate)
            or estimate <= 0
        ):
            raise ValueError(
                f"{benchmark['data']}: a measured workload estimate is required"
            )
        resource = spec["resource_profiles"].get(benchmark.get("resource_profile"), {})
        if any(not resource.get(field) for field in ("gpu", "time", "mem", "evidence")):
            raise ValueError(
                f"{benchmark['data']}: explicit measured resource requests are required"
            )
        if not re.fullmatch(r"[A-Za-z0-9_]+:1", resource["gpu"]):
            raise ValueError("resource requests must specify exactly one typed GPU")
    dependency = spec.get("training_job_id")
    if not dependency or not str(dependency).isdigit():
        raise ValueError("a verified historical training dependency is required")
    completed_approval = spec.get("completed_checkpoint_approval")
    if completed_approval is not None and (
        not isinstance(completed_approval, dict)
        or str(completed_approval.get("job_id")) != str(dependency)
        or any(
            not isinstance(completed_approval.get(field), str)
            or not completed_approval[field].strip()
            for field in ("approved_at", "reference")
        )
    ):
        raise ValueError(
            "completed_checkpoint_approval needs the matching job_id, approved_at, "
            "and an archived approval reference"
        )
    accounts = sorted(spec["accounts"], key=lambda account: account["alias"])
    if (
        len(accounts) != 3
        or {account["user"] for account in accounts}
        != {"danieloh", "guyzagor", "odedshah"}
        or len({account["alias"] for account in accounts}) != 3
    ):
        raise ValueError("all three approved accounts must be present exactly once")
    for account in accounts:
        for field in ("project_dir", "venv", "checkpoint_path"):
            value = account.get(field, "")
            path = Path(value)
            if (
                not path.is_absolute()
                or str(path) == "/"
                or ".." in path.parts
                or any(character in value for character in ",\n\r")
                or any(
                    path == Path(root) or Path(root) in path.parents
                    for root in ("/tmp", "/scratch", "/var/tmp")
                )
            ):
                raise ValueError(
                    f"{account['alias']}: {field} needs an absolute durable path"
                )
        for field in ("checkpoint_sha256", "checkpoint_config", "code", "environment"):
            if (
                not spec["expected"].get(field)
                or account.get(field) != spec["expected"][field]
            ):
                raise ValueError(
                    f"{account['alias']}: unverified or mismatched {field}"
                )
        if account.get("durable_checkpoint") is not True:
            raise ValueError(f"{account['alias']}: durable_checkpoint is not verified")
        proof = account.get("live_proof", {})
        if (
            not proof.get("checked_at")
            or not proof.get("evidence")
            or proof.get("gpu_slots", 0) < 1
        ):
            raise ValueError(
                f"{account['alias']}: live_proof needs current evidence and available GPU slots"
            )
        if completed_approval is None and str(proof.get("training_job_id")) != str(
            dependency
        ):
            raise ValueError(
                f"{account['alias']}: historical training dependency is unresolved"
            )
        for resource in spec["resource_profiles"].values():
            if resource["gpu"] not in proof.get("gpus", []):
                raise ValueError(
                    f"{account['alias']}: resource GPU has no verified entitlement"
                )
    loads = {
        account["alias"]: [0.0] * account["live_proof"]["gpu_slots"]
        for account in accounts
    }
    tiers = [
        ["scbench_kv"],
        [
            name
            for name in inventory
            if name.startswith("ruler_") and name.endswith("_4k")
        ],
        [
            name
            for name in inventory
            if name.startswith("ruler_") and name.endswith("_8k")
        ],
    ]
    priority = {name for tier in tiers for name in tier}
    tiers.append([name for name in inventory if name not in priority])
    rows = []
    for tier_index, names in enumerate(tiers):
        for name in sorted(
            names, key=lambda name: (-benchmarks[name]["estimated_seconds"], name)
        ):
            account = min(
                accounts,
                key=lambda account: (
                    min(loads[account["alias"]]),
                    sum(loads[account["alias"]]) / len(loads[account["alias"]]),
                    account["alias"],
                ),
            )
            slots = loads[account["alias"]]
            slots[slots.index(min(slots))] += benchmarks[name]["estimated_seconds"]
            project = account["project_dir"]
            run_name = f"{spec['name']}-{name}"
            run_dir = f"{project}/results/{run_name}"
            if run_dir in account["live_proof"].get("active_result_dirs", []):
                raise ValueError(f"{run_dir}: an existing writer must be reconciled")
            resource = spec["resource_profiles"][benchmarks[name]["resource_profile"]]
            command = [
                "sbatch",
                "--parsable",
                f"--chdir={project}",
                f"--job-name={run_name}",
                f"--output={project}/.slurm/logs/%j-%x.log",
                f"--gpus={resource['gpu']}",
                f"--time={resource['time']}",
                f"--mem={resource['mem']}",
                f"--export=ALL,FASTKVZIP_VENV={account['venv']},EVAL_GRAPH_SCRIPT=prefill/eval_graph.py",
            ]
            if completed_approval is None:
                command.append(f"--dependency=afterok:{dependency}")
            if resource.get("tmp"):
                command.append(f"--tmp={resource['tmp']}")
            command.extend(
                [
                    f"{project}/slurm/eval_graph.sbatch",
                    run_name,
                    "--graph-checkpoint",
                    account["checkpoint_path"],
                    "--data",
                    name,
                    "--level",
                    "pair",
                    "--window-size",
                    "0",
                    "--ratios",
                    "0.75",
                    "0.50",
                    "0.40",
                    "0.30",
                    "0.20",
                    "--full-cache-answer",
                    "--existing-results",
                    "resume",
                    "--token-microbatch-size",
                    str(spec["token_microbatch_size"]),
                    "--graph-microbatch-size",
                    str(spec["graph_microbatch_size"]),
                    "--ruler-prompt-mode",
                    "graphkv",
                    "--wandb-run-id",
                    spec["wandb"]["run_id"],
                ]
            )
            rows.append(
                {
                    **benchmarks[name],
                    "tier": tier_index,
                    "alias": account["alias"],
                    "user": account["user"],
                    "project_dir": project,
                    "run_name": run_name,
                    "run_dir": run_dir,
                    "command": command,
                }
            )
    manifest = {**copy.deepcopy(spec), "rows": rows}
    manifest["sha256"] = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return manifest


def next_submission(manifest, receipt, *, approved_sha256):
    """Return the next unsubmitted row, without mutating receipts or submitting.

    Receipts contain ``manifest_sha256`` and append-only ``attempts`` entries with
    ``data``, ``status`` and ``job_id``. Known failed jobs are not auto-retried.
    An ambiguous response may become ``not-submitted`` only with recorded
    ``reconciliation_evidence`` that no job was created. Keep the original error.
    """
    actual = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if not approved_sha256 or not actual == manifest.get(
        "sha256"
    ) == approved_sha256 == receipt.get("manifest_sha256"):
        raise ValueError("exact manifest approval and matching receipt are required")
    seen, active = set(), set()
    for attempt in receipt["attempts"]:
        if attempt["status"] == "not-submitted" and attempt.get(
            "reconciliation_evidence"
        ):
            continue
        if (
            attempt["status"] in {"submitting", "unknown"}
            or not str(attempt.get("job_id", "")).isdigit()
        ):
            raise ValueError(
                "uncertain submission outcome: reconcile before any further submission"
            )
        name = attempt["data"]
        seen.add(name)
        if attempt["status"] in {"submitted", "pending", "running"}:
            if name in active:
                raise ValueError(f"{name}: more than one possible result writer")
            active.add(name)
    ordered = [row["data"] for row in manifest["rows"]]
    if seen != set(ordered[: len(seen)]):
        raise ValueError("receipt does not follow the approved submission order")
    return (
        copy.deepcopy(manifest["rows"][len(seen)]) if len(seen) < len(ordered) else None
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print only (also the default; this helper never submits)",
    )
    parser.parse_args()
    try:
        print(
            json.dumps(build_manifest(json.load(sys.stdin)), indent=2, allow_nan=False)
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
