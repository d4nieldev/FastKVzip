"""Production planning stays offline and uses measured, verified inputs."""

import copy
import runpy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _planner():
    return runpy.run_path(str(ROOT / "slurm" / "plan_ruler_evaluation.py"))


def _spec():
    names = runpy.run_path(str(ROOT / "prefill/data/benchmarks.py"))["get_data_list"](
        "all"
    )
    expected = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_config": {
            "model": "Qwen/Qwen2.5-7B-Instruct-1M",
            "objective": "answer",
        },
        "code": {"commit": "b" * 40},
        "environment": {"torch": "2.7.0", "datasets": "4.0.0"},
    }
    accounts = []
    for alias, user, slots in (
        ("bgu-slurm", "danieloh", 1),
        ("bgu-slurm-guyzagor", "guyzagor", 2),
        ("bgu-slurm-odedshah", "odedshah", 3),
    ):
        accounts.append(
            {
                "alias": alias,
                "user": user,
                "project_dir": f"/home/{user}/FastKVzip",
                "venv": f"/home/{user}/venv",
                "checkpoint_path": f"/home/{user}/checkpoints/uniform/best.pt",
                **copy.deepcopy(expected),
                "durable_checkpoint": True,
                "live_proof": {
                    "checked_at": "2026-09-08T10:00:00+00:00",
                    "evidence": f"preflight-{user}.json",
                    "gpu_slots": slots,
                    "gpus": ["rtx_pro_6000:1"],
                    "active_result_dirs": [],
                    "training_job_id": "21061828",
                },
            }
        )
    return {
        "name": "uniform-eval-all-w0",
        "expected": expected,
        "wandb": {
            "entity": "team",
            "project": "existing-project",
            "run_id": "new-evaluation-run",
        },
        "source_wandb_run_id": "original-training-run",
        "training_job_id": "21061828",
        "accounts": accounts,
        "resource_profiles": {
            "measured": {
                "gpu": "rtx_pro_6000:1",
                "time": "02:00:00",
                "mem": "60G",
                "evidence": "pilot-measurements.json",
            },
        },
        "token_microbatch_size": 16000,
        "graph_microbatch_size": 16,
        "benchmarks": [
            {"data": name, "estimated_seconds": 100, "resource_profile": "measured"}
            for name in reversed(names)
        ],
    }


def test_manifest_contains_107_unique_benchmarks_in_approved_submission_tiers():
    manifest = _planner()["build_manifest"](_spec())
    rows = manifest["rows"]
    assert len(rows) == len({row["data"] for row in rows}) == 107
    assert rows[0]["data"] == "scbench_kv"
    assert [row["tier"] for row in rows] == [0] + [1] * 13 + [2] * 13 + [3] * 80
    assert all(
        row["data"].startswith("ruler_") and row["data"].endswith("_4k")
        for row in rows[1:14]
    )
    assert all(
        row["data"].startswith("ruler_") and row["data"].endswith("_8k")
        for row in rows[14:27]
    )
    assert "agentic" not in {row["data"] for row in rows}
    assert {row["user"] for row in rows} == {"danieloh", "guyzagor", "odedshah"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "c" * 64),
        ("checkpoint_config", {"model": "wrong-model"}),
        ("code", {"commit": "c" * 40}),
        ("environment", {"torch": "wrong-version"}),
        ("durable_checkpoint", False),
        ("live_proof", {}),
    ],
)
def test_every_account_must_pass_checkpoint_and_live_preflight_before_any_row(
    field, value
):
    spec = _spec()
    spec["accounts"][-1][field] = value
    with pytest.raises(ValueError, match=field):
        _planner()["build_manifest"](spec)


@pytest.mark.parametrize(
    "change",
    [
        "missing-account",
        "duplicate-account",
        "missing-benchmark",
        "duplicate-benchmark",
        "agentic",
    ],
)
def test_manifest_refuses_any_change_to_approved_accounts_or_inventory(change):
    spec = _spec()
    if change == "missing-account":
        spec["accounts"].pop()
    elif change == "duplicate-account":
        spec["accounts"][-1] = copy.deepcopy(spec["accounts"][0])
    elif change == "missing-benchmark":
        spec["benchmarks"].pop()
    elif change == "duplicate-benchmark":
        spec["benchmarks"].append(copy.deepcopy(spec["benchmarks"][0]))
    else:
        spec["benchmarks"].append(
            {"data": "agentic", "estimated_seconds": 1, "resource_profile": "measured"}
        )
    with pytest.raises(ValueError, match="accounts|inventory"):
        _planner()["build_manifest"](spec)


def test_commands_use_standard_wrapper_full_datasets_and_one_bound_nonlogging_destination():
    manifest = _planner()["build_manifest"](_spec())
    assert manifest["wandb"]["run_id"] == "new-evaluation-run"
    assert len(manifest["sha256"]) == 64
    for row in manifest["rows"]:
        command = row["command"]
        assert command[:2] == ["sbatch", "--parsable"]
        assert "--gpus=rtx_pro_6000:1" in command
        assert "--time=02:00:00" in command and "--mem=60G" in command
        assert [
            argument for argument in command if argument.startswith("--dependency=")
        ] == ["--dependency=afterok:21061828"]
        assert f"{row['project_dir']}/slurm/eval_graph.sbatch" in command
        assert command[command.index("--window-size") + 1] == "0"
        assert command[command.index("--level") + 1] == "pair"
        assert command[
            command.index("--ratios") + 1 : command.index("--ratios") + 6
        ] == ["0.75", "0.50", "0.40", "0.30", "0.20"]
        assert command[command.index("--existing-results") + 1] == "resume"
        assert command[command.index("--wandb-run-id") + 1] == "new-evaluation-run"
        assert command[command.index("--ruler-prompt-mode") + 1] == "graphkv"
        assert "--full-cache-answer" in command
        assert not {
            "--num",
            "--log-to-wandb",
            "--wandb-project",
            "--wandb-entity",
            "--ruler-data-dir",
        }.intersection(command)
        assert row["run_dir"] == f"{row['project_dir']}/results/{row['run_name']}"
    assert len({row["run_dir"] for row in manifest["rows"]}) == 107


@pytest.mark.parametrize("field", ["gpu", "time", "mem", "evidence"])
def test_resources_cannot_be_invented_when_measurements_are_missing(field):
    spec = _spec()
    del spec["resource_profiles"]["measured"][field]
    with pytest.raises(ValueError, match="resource"):
        _planner()["build_manifest"](spec)


@pytest.mark.parametrize(
    "problem",
    [
        "multiple-gpus",
        "unavailable-gpu",
        "unknown-dependency",
        "active-writer",
        "missing-estimate",
    ],
)
def test_manifest_stops_for_unverified_production_requests(problem):
    spec = _spec()
    if problem == "multiple-gpus":
        spec["resource_profiles"]["measured"]["gpu"] = "rtx_pro_6000:2"
    elif problem == "unavailable-gpu":
        spec["accounts"][0]["live_proof"]["gpus"] = []
    elif problem == "unknown-dependency":
        spec["accounts"][0]["live_proof"]["training_job_id"] = None
    elif problem == "active-writer":
        spec["accounts"][0]["live_proof"]["active_result_dirs"] = [
            "/home/danieloh/FastKVzip/results/uniform-eval-all-w0-scbench_kv"
        ]
    else:
        spec["benchmarks"][0]["estimated_seconds"] = None
    with pytest.raises(ValueError, match="resource|dependency|writer|estimate"):
        _planner()["build_manifest"](spec)


def test_archived_completed_checkpoint_approval_omits_only_the_expired_dependency():
    spec = _spec()
    for account in spec["accounts"]:
        account["live_proof"]["training_job_id"] = None
    spec["completed_checkpoint_approval"] = {
        "job_id": "21061828",
        "approved_at": "2026-09-08T12:00:00+00:00",
        "reference": "user-approval-completed-checkpoint.json",
        "reason": "Training completed; the scheduler no longer retains its job.",
    }
    manifest = _planner()["build_manifest"](spec)
    assert manifest["training_job_id"] == "21061828"
    assert (
        manifest["completed_checkpoint_approval"]
        == spec["completed_checkpoint_approval"]
    )
    assert len(manifest["rows"]) == 107
    assert not any(
        argument.startswith("--dependency")
        for row in manifest["rows"]
        for argument in row["command"]
    )
    for row in manifest["rows"]:
        assert "--graph-checkpoint" in row["command"]
        assert (
            row["command"][row["command"].index("--existing-results") + 1] == "resume"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "wrong-job"),
        ("approved_at", None),
        ("approved_at", "  "),
        ("reference", ""),
        ("reference", True),
    ],
)
def test_completed_checkpoint_exception_requires_a_matching_archived_approval(
    field, value
):
    spec = _spec()
    spec["completed_checkpoint_approval"] = {
        "job_id": "21061828",
        "approved_at": "2026-09-08T12:00:00+00:00",
        "reference": "user-approval-completed-checkpoint.json",
        field: value,
    }
    with pytest.raises(ValueError, match="completed_checkpoint_approval"):
        _planner()["build_manifest"](spec)


@pytest.mark.parametrize("approval", [True, {}, []])
def test_a_boolean_or_empty_record_cannot_remove_the_scheduler_dependency(approval):
    spec = _spec()
    spec["completed_checkpoint_approval"] = approval
    with pytest.raises(ValueError, match="completed_checkpoint_approval"):
        _planner()["build_manifest"](spec)


@pytest.mark.parametrize("account_index", [0, 1, 2])
@pytest.mark.parametrize("field", ["checkpoint_sha256", "checkpoint_config"])
def test_completed_checkpoint_approval_never_weakens_any_account_checkpoint_gate(
    account_index, field
):
    spec = _spec()
    spec["completed_checkpoint_approval"] = {
        "job_id": "21061828",
        "approved_at": "2026-09-08T12:00:00+00:00",
        "reference": "user-approval-completed-checkpoint.json",
    }
    for account in spec["accounts"]:
        account["live_proof"]["training_job_id"] = None
    del spec["accounts"][account_index][field]
    with pytest.raises(ValueError, match=field):
        _planner()["build_manifest"](spec)


def test_expensive_work_is_spread_before_smaller_jobs_and_balanced_by_live_slots():
    spec = _spec()
    manifest = _planner()["build_manifest"](spec)
    loads = {"danieloh": 0, "guyzagor": 0, "odedshah": 0}
    for row in manifest["rows"]:
        loads[row["user"]] += row["estimated_seconds"]
    normalized = [loads["danieloh"], loads["guyzagor"] / 2, loads["odedshah"] / 3]
    assert max(normalized) - min(normalized) <= 100

    for account in spec["accounts"]:
        account["live_proof"]["gpu_slots"] = 1
    estimates = {
        "ruler_niah_single_1_4k": 900,
        "ruler_niah_single_2_4k": 800,
        "ruler_niah_single_3_4k": 700,
    }
    for benchmark in spec["benchmarks"]:
        benchmark["estimated_seconds"] = estimates.get(benchmark["data"], 100)
    rows = _planner()["build_manifest"](spec)["rows"]
    assert [row["estimated_seconds"] for row in rows[1:4]] == [900, 800, 700]
    assert len({row["alias"] for row in rows[1:4]}) == 3


def test_submission_preview_requires_exact_approval_and_preserves_attempt_history():
    planner = _planner()
    manifest = planner["build_manifest"](_spec())
    receipt = {"manifest_sha256": manifest["sha256"], "attempts": []}
    with pytest.raises(ValueError, match="approval"):
        planner["next_submission"](manifest, receipt, approved_sha256=None)
    first = planner["next_submission"](
        manifest, receipt, approved_sha256=manifest["sha256"]
    )
    assert first["data"] == "scbench_kv"

    receipt["attempts"].extend(
        [
            {"data": "scbench_kv", "status": "failed", "job_id": "111"},
            {"data": "scbench_kv", "status": "submitted", "job_id": "112"},
        ]
    )
    before = copy.deepcopy(receipt)
    second = planner["next_submission"](
        manifest, receipt, approved_sha256=manifest["sha256"]
    )
    assert second == manifest["rows"][1]
    assert receipt == before
    assert "--dependency=afterok:21061828" in second["command"]
    assert not any(
        "111" in argument or "112" in argument for argument in second["command"]
    )


@pytest.mark.parametrize("status", ["submitting", "unknown"])
def test_uncertain_sbatch_outcome_stops_the_entire_grid_until_reconciled(status):
    planner = _planner()
    manifest = planner["build_manifest"](_spec())
    receipt = {
        "manifest_sha256": manifest["sha256"],
        "attempts": [
            {"data": "scbench_kv", "status": status, "job_id": None},
        ],
    }
    with pytest.raises(ValueError, match="uncertain"):
        planner["next_submission"](
            manifest, receipt, approved_sha256=manifest["sha256"]
        )


def test_changed_command_after_approval_or_two_active_attempts_is_rejected():
    planner = _planner()
    manifest = planner["build_manifest"](_spec())
    receipt = {"manifest_sha256": manifest["sha256"], "attempts": []}
    changed = copy.deepcopy(manifest)
    changed["rows"][0]["command"].append("--log-to-wandb")
    with pytest.raises(ValueError, match="approval"):
        planner["next_submission"](changed, receipt, approved_sha256=manifest["sha256"])

    receipt["attempts"] = [
        {"data": "scbench_kv", "status": "running", "job_id": "111"},
        {"data": "scbench_kv", "status": "submitted", "job_id": "112"},
    ]
    with pytest.raises(ValueError, match="writer"):
        planner["next_submission"](
            manifest, receipt, approved_sha256=manifest["sha256"]
        )


def test_cli_dry_run_prints_manifest_without_scheduler_or_heavy_model_imports():
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "slurm/plan_ruler_evaluation.py"),
            "--dry-run",
        ],
        input=json.dumps(_spec()),
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stderr
    assert len(json.loads(completed.stdout)["rows"]) == 107


def test_production_cannot_bind_to_the_source_training_run():
    spec = _spec()
    spec["wandb"]["run_id"] = spec["source_wandb_run_id"]
    with pytest.raises(ValueError, match="new evaluation"):
        _planner()["build_manifest"](spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_path", "relative/best.pt"),
        ("checkpoint_path", "/scratch/job/best.pt"),
        ("project_dir", "/tmp/transient-project"),
        ("venv", "/home/user/venv,UNSAFE=1"),
    ],
)
def test_remote_paths_must_be_absolute_durable_and_safe_for_sbatch_export(field, value):
    spec = _spec()
    spec["accounts"][0][field] = value
    with pytest.raises(ValueError, match="path"):
        _planner()["build_manifest"](spec)


def test_one_long_job_does_not_hide_the_same_accounts_other_idle_slots():
    spec = _spec()
    for account in spec["accounts"]:
        account["live_proof"]["gpu_slots"] = 5
    next(row for row in spec["benchmarks"] if row["data"] == "scbench_kv")[
        "estimated_seconds"
    ] = 1_000_000
    rows = _planner()["build_manifest"](spec)["rows"]
    assert {row["user"] for row in rows[1:14]} == {"danieloh", "guyzagor", "odedshah"}


def test_missing_checkpoint_path_blocks_before_building_any_commands():
    spec = _spec()
    del spec["accounts"][-1]["checkpoint_path"]
    with pytest.raises(ValueError, match="checkpoint_path"):
        _planner()["build_manifest"](spec)


def test_confirmed_non_submission_can_be_retried_without_erasing_the_uncertain_attempt():
    planner = _planner()
    manifest = planner["build_manifest"](_spec())
    receipt = {
        "manifest_sha256": manifest["sha256"],
        "attempts": [
            {
                "data": "scbench_kv",
                "status": "not-submitted",
                "job_id": None,
                "submission_error": "SSH response lost",
                "reconciliation_evidence": "squeue-sacct-bounded-recheck-no-job.json",
            }
        ],
    }
    before = copy.deepcopy(receipt)
    assert (
        planner["next_submission"](
            manifest, receipt, approved_sha256=manifest["sha256"]
        )["data"]
        == "scbench_kv"
    )
    assert receipt == before
    del receipt["attempts"][0]["reconciliation_evidence"]
    with pytest.raises(ValueError, match="uncertain"):
        planner["next_submission"](
            manifest, receipt, approved_sha256=manifest["sha256"]
        )
