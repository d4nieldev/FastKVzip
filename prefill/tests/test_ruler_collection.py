"""Collection follows approved attempt locations, never a moving account default."""

import copy
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess

import pytest

from results.evaluation_run import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]


def collector():
    return runpy.run_path(str(ROOT / "slurm/collect_ruler_evaluation.py"))["run"]


def stamp(payload):
    payload = copy.deepcopy(payload)
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return payload


def grid(directory):
    rows = [
        {
            "data": data,
            "alias": "source",
            "user": "alice",
            "run_name": f"grid-{data}",
            "run_dir": f"/home/alice/results/grid-{data}",
        }
        for data in ("scbench_kv", "ruler_niah_single_1_4k")
    ]
    manifest = stamp(
        {
            "rows": rows,
            "wandb": {
                "entity": "team",
                "project": "project",
                "run_id": "evaluation-run",
            },
            "expected": {"checkpoint_sha256": "a" * 64},
        }
    )
    receipt = {
        "manifest_sha256": manifest["sha256"],
        "attempts": [
            {**row, "status": "submitted", "job_id": str(job)}
            for row, job in zip(rows, (101, 102))
        ],
    }
    atomic_write_json(directory / "manifest.json", manifest)
    atomic_write_json(directory / "receipt.json", receipt)
    return manifest, receipt


def boundary(calls, *, failed="102", live=""):
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh":
            remote = shlex.split(command[-1])
            ids = next(
                arg.split("=", 1)[1] for arg in remote if arg.startswith("--jobs=")
            ).split(",")
            if remote[0] == "squeue":
                output = "".join(f"{job}|RUNNING|node\n" for job in ids if job == live)
            else:
                output = "".join(
                    f"{job}|{'FAILED|1:0' if job == failed else 'COMPLETED|0:0'}\n"
                    for job in ids
                )
            return subprocess.CompletedProcess(command, 0, output, "")
        assert command[0] == "rsync"
        assert "--delete" not in command and "--inplace" not in command
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_existing_grid_collects_once_and_preserves_upload_identity(tmp_path):
    manifest, _ = grid(tmp_path)
    calls, uploads = [], []

    def upload(paths, **kwargs):
        uploads.append((paths, kwargs))
        return {"uploaded_points": 6}

    run = collector()
    result = run(
        tmp_path,
        approved_sha256=manifest["sha256"],
        collect=True,
        runner=boundary(calls),
        uploader=upload,
    )
    assert result["collected"] == 1
    assert uploads == [
        (
            [tmp_path / "production/alice/grid-scbench_kv"],
            {"wandb_run_id": "evaluation-run", "project": "project", "entity": "team"},
        )
    ]
    run(
        tmp_path,
        approved_sha256=manifest["sha256"],
        collect=True,
        runner=boundary(calls),
        uploader=upload,
    )
    assert len(uploads) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "approval",
        "parent",
        "wandb",
        "checkpoint",
        "partial",
        "name",
        "owner",
        "duplicate-job",
    ],
)
def test_unsafe_retry_is_rejected_before_network_or_upload(tmp_path, problem):
    manifest, receipt = grid(tmp_path)
    retry, retry_receipt = retry_grid(tmp_path, manifest, receipt)
    if problem == "parent":
        retry["source_manifest_sha256"] = "wrong"
    elif problem == "wandb":
        retry["wandb"]["run_id"] = "training-run"
    elif problem == "checkpoint":
        retry["expected"]["checkpoint_sha256"] = "b" * 64
    elif problem == "partial":
        retry["rows"][0]["source_output_count"] = 5
    elif problem == "name":
        retry["rows"][0]["run_name"] = "different-run"
    elif problem == "owner":
        retry_receipt["attempts"][0]["alias"] = "unapproved"
    elif problem == "duplicate-job":
        retry_receipt["attempts"][0]["job_id"] = "101"
    retry = stamp({key: value for key, value in retry.items() if key != "sha256"})
    retry_receipt["manifest_sha256"] = retry["sha256"]
    atomic_write_json(tmp_path / "retry-manifest.json", retry)
    atomic_write_json(tmp_path / "retry-receipt.json", retry_receipt)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid approval must stop before external calls")

    with pytest.raises(ValueError):
        collector()(
            tmp_path,
            approved_sha256=manifest["sha256"],
            retry_approved_sha256=None if problem == "approval" else retry["sha256"],
            collect=True,
            runner=forbidden,
            uploader=forbidden,
        )


@pytest.mark.parametrize("blocked", ["old-running", "new-running", "unknown"])
def test_no_collection_while_either_attempt_may_be_a_writer(tmp_path, blocked):
    manifest, receipt = grid(tmp_path)
    retry, retry_receipt = retry_grid(tmp_path, manifest, receipt)
    if blocked == "unknown":
        retry_receipt["attempts"][0]["status"] = "unknown"
        retry_receipt["attempts"][0].pop("job_id")
        atomic_write_json(tmp_path / "retry-receipt.json", retry_receipt)
    calls = []
    live = {"old-running": "102", "new-running": "103"}.get(blocked, "")
    result = collector()(
        tmp_path,
        approved_sha256=manifest["sha256"],
        retry_approved_sha256=retry["sha256"],
        collect=True,
        runner=boundary(calls, live=live),
        uploader=lambda *a, **k: {},
    )
    assert result["collected"] == 1
    assert all(
        "destination:" not in arg
        for call in calls
        if call[0] == "rsync"
        for arg in call
    )


@pytest.mark.parametrize("new_retry", [False, True])
def test_a_new_attempt_during_copy_is_not_marked_collected(tmp_path, new_retry):
    manifest, receipt = grid(tmp_path)
    calls = []
    original = boundary(calls)

    def runner(command, **kwargs):
        result = original(command, **kwargs)
        if command[0] == "rsync":
            if new_retry:
                retry_grid(tmp_path, manifest, receipt)
            else:
                receipt["attempts"].append({**receipt["attempts"][0], "job_id": "104"})
                atomic_write_json(tmp_path / "receipt.json", receipt)
        return result

    with pytest.raises(ValueError, match="receipt changed"):
        collector()(
            tmp_path,
            approved_sha256=manifest["sha256"],
            collect=True,
            runner=runner,
            uploader=lambda *a, **k: pytest.fail("must not upload"),
        )
    assert not (tmp_path / "collection.json").exists()


def test_dry_run_is_local_only_and_retry_upload_failure_is_resumable(tmp_path):
    manifest, receipt = grid(tmp_path)
    retry, _ = retry_grid(tmp_path, manifest, receipt)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run may not call external services")

    kwargs = {
        "approved_sha256": manifest["sha256"],
        "retry_approved_sha256": retry["sha256"],
    }
    before = set(tmp_path.iterdir())
    assert collector()(tmp_path, **kwargs, runner=forbidden, uploader=forbidden)[
        "dry_run"
    ]
    assert set(tmp_path.iterdir()) == before
    calls = []

    def fail_upload(*args, **kwargs):
        raise RuntimeError("upload interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        collector()(
            tmp_path,
            **kwargs,
            collect=True,
            runner=boundary(calls),
            uploader=fail_upload,
        )
    assert json.loads((tmp_path / "collection.json").read_text())["upload_pending"]
    uploads = []
    collector()(
        tmp_path,
        **kwargs,
        collect=True,
        runner=boundary(calls),
        uploader=lambda *a, **k: uploads.append(k) or {},
    )
    assert len(uploads) == 1
    assert len([call for call in calls if call[0] == "rsync"]) == 2


def test_pending_upload_excludes_snapshots_with_a_new_active_attempt(tmp_path):
    manifest, receipt = grid(tmp_path)
    run = collector()
    calls = []

    def fail_upload(*args, **kwargs):
        raise RuntimeError("upload interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run(
            tmp_path,
            approved_sha256=manifest["sha256"],
            collect=True,
            runner=boundary(calls, failed=""),
            uploader=fail_upload,
        )
    retry, _ = retry_grid(tmp_path, manifest, receipt)
    uploads = []
    kwargs = {
        "approved_sha256": manifest["sha256"],
        "retry_approved_sha256": retry["sha256"],
        "collect": True,
        "uploader": lambda paths, **kw: uploads.append(paths) or {},
    }
    run(tmp_path, **kwargs, runner=boundary(calls, failed="", live="103"))
    assert uploads == [[tmp_path / "production/alice/grid-scbench_kv"]]
    assert json.loads((tmp_path / "collection.json").read_text())["upload_pending"]
    run(tmp_path, **kwargs, runner=boundary(calls, failed=""))
    assert uploads[-1] == [
        tmp_path / "production/alice/grid-scbench_kv",
        tmp_path / "production/bob/grid-ruler_niah_single_1_4k",
    ]
    assert not json.loads((tmp_path / "collection.json").read_text())["upload_pending"]


def retry_grid(directory, manifest, receipt):
    row = {
        **manifest["rows"][1],
        "alias": "destination",
        "user": "bob",
        "run_dir": "/home/bob/results/grid-ruler_niah_single_1_4k",
        "source_job_id": "102",
        "source_output_count": 0,
    }
    retry = stamp(
        {
            "source_manifest_sha256": manifest["sha256"],
            "wandb": manifest["wandb"],
            "expected": manifest["expected"],
            "rows": [row],
        }
    )
    retry_receipt = {
        "manifest_sha256": retry["sha256"],
        "attempts": [{**row, "status": "submitted", "job_id": "103"}],
    }
    atomic_write_json(directory / "retry-manifest.json", retry)
    atomic_write_json(directory / "retry-receipt.json", retry_receipt)
    return retry, retry_receipt


def test_moved_retry_collects_from_approved_new_account_and_keeps_original_history(
    tmp_path,
):
    manifest, receipt = grid(tmp_path)
    retry, _ = retry_grid(tmp_path, manifest, receipt)
    calls, uploads = [], []

    def upload(paths, **kwargs):
        uploads.append((paths, kwargs))
        return {"uploaded_points": 12}

    run = collector()
    result = run(
        tmp_path,
        approved_sha256=manifest["sha256"],
        retry_approved_sha256=retry["sha256"],
        collect=True,
        runner=boundary(calls),
        uploader=upload,
    )
    assert result["collected"] == 2
    assert (
        result["failed_jobs"] == []
    )  # Old failures remain history, not unresolved work.
    assert [row["job_id"] for row in result["failed_attempts"]] == ["102"]
    copies = [call for call in calls if call[0] == "rsync"]
    assert (
        copies[-1][-2] == "destination:/home/bob/results/grid-ruler_niah_single_1_4k/"
    )
    assert uploads[0][1]["wandb_run_id"] == "evaluation-run"
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    run(
        tmp_path,
        approved_sha256=manifest["sha256"],
        retry_approved_sha256=retry["sha256"],
        collect=True,
        runner=boundary(calls),
        uploader=upload,
    )
    assert len(uploads) == 1
