"""One collection pass; default dry-run is local-only. Run --collect after the VPN gate."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prefill"))
from results.evaluation_run import atomic_write_json

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ConnectionAttempts=1",
]


def eligible(attempts, jobs):
    return (
        bool(attempts)
        and all(
            a["status"] not in {"submitting", "unknown"}
            and not jobs.get(str(a.get("job_id")), {}).get("live", False)
            and jobs.get(str(a.get("job_id")), {}).get("state") in TERMINAL
            for a in attempts
        )
        and jobs[str(attempts[-1]["job_id"])]["state"] == "COMPLETED"
        and jobs[str(attempts[-1]["job_id"])]["exit_code"] == "0:0"
    )


def poll(alias, ids, runner):
    jobs = {job: {"state": "UNKNOWN", "exit_code": None, "live": False} for job in ids}
    live = {}
    for command in (
        ["squeue", "--noheader", f"--jobs={','.join(ids)}", "--format=%i|%T|%R"],
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            f"--jobs={','.join(ids)}",
            "--format=JobIDRaw,State%30,ExitCode",
        ],
    ):
        try:
            result = runner(
                [*SSH, alias, shlex.join(command)],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.CalledProcessError as error:
            if command[0] == "squeue" and "Invalid job id specified" in (
                error.stderr or ""
            ):
                continue
            raise RuntimeError(
                f"{alias}: scheduler/connection check failed; return to the VPN gate"
            ) from error
        for line in result.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 3 or fields[0] not in jobs:
                continue
            job, state, detail = fields[:3]
            state = state.split()[0].rstrip("+")
            if command[0] == "squeue":
                live[job] = {
                    "state": state,
                    "reason": detail,
                    "live": True,
                    "exit_code": None,
                }
            else:
                jobs[job] = {"state": state, "exit_code": detail, "live": False}
    jobs.update(live)  # Live queue evidence wins over lagging accounting.
    return jobs


def run(
    directory,
    *,
    approved_sha256,
    retry_approved_sha256=None,
    collect=False,
    runner=subprocess.run,
    uploader=None,
):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    receipt = json.loads((directory / "receipt.json").read_text())
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if (
        not digest
        == manifest.get("sha256")
        == receipt.get("manifest_sha256")
        == approved_sha256
    ):
        raise ValueError("exact approved manifest and matching receipt are required")
    rows = {row["data"]: row for row in manifest["rows"]}
    plans = [(rows, receipt)]
    snapshots = {
        directory / "receipt.json": receipt,
        directory / "retry-receipt.json": None,
    }
    if (directory / "retry-receipt.json").exists() or retry_approved_sha256:
        retry = json.loads((directory / "retry-manifest.json").read_text())
        retry_receipt = json.loads((directory / "retry-receipt.json").read_text())
        retry_digest = hashlib.sha256(
            json.dumps(
                {k: v for k, v in retry.items() if k != "sha256"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if (
            not retry_approved_sha256
            or not retry_digest
            == retry.get("sha256")
            == retry_receipt.get("manifest_sha256")
            == retry_approved_sha256
        ):
            raise ValueError(
                "exact retry manifest approval and matching receipt are required"
            )
        if (
            retry.get("source_manifest_sha256") != digest
            or retry.get("wandb") != manifest["wandb"]
            or retry.get("expected", {}).get("checkpoint_sha256")
            != manifest["expected"]["checkpoint_sha256"]
        ):
            raise ValueError(
                "retry must preserve the original grid, checkpoint and W&B identity"
            )
        retry_rows = {row["data"]: row for row in retry["rows"]}
        if (
            len(retry_rows) != len(retry["rows"])
            or not retry_rows.keys() <= rows.keys()
        ):
            raise ValueError("retry rows must be a unique subset of the original grid")
        for data, row in retry_rows.items():
            original = rows[data]
            source = [
                a for a in receipt["attempts"] if a["data"] == data and a.get("job_id")
            ]
            if (
                not source
                or str(row.get("source_job_id")) != str(source[-1]["job_id"])
                or row["run_name"] != original["run_name"]
            ):
                raise ValueError("retry must name its original attempt and logical run")
            if row["alias"] != original["alias"]:
                if (
                    type(row.get("source_output_count")) is not int
                    or row["source_output_count"] != 0
                ):
                    raise ValueError(
                        "partial outputs must resume on their original account"
                    )
            elif (
                row["user"] != original["user"] or row["run_dir"] != original["run_dir"]
            ):
                raise ValueError(
                    "same-account retries must preserve the result directory"
                )
        plans.append((retry_rows, retry_receipt))
        snapshots[directory / "retry-receipt.json"] = retry_receipt

    def check_receipts(phase):
        if any(
            (json.loads(path.read_text()) if path.exists() else None) != saved
            for path, saved in snapshots.items()
        ):
            raise ValueError(
                f"receipt changed during {phase}; reconcile before collection"
            )

    attempts, placements = [], {}
    for plan_rows, plan_receipt in plans:
        for attempt in plan_receipt["attempts"]:
            row = plan_rows.get(attempt.get("data"))
            if row is None or any(
                attempt.get(field) != row[field]
                for field in ("alias", "user", "run_name", "run_dir")
            ):
                raise ValueError(
                    "receipt owner/data does not match the approved manifest"
                )
            if attempt["status"] == "not-submitted" and attempt.get(
                "reconciliation_evidence"
            ):
                continue
            job = str(attempt.get("job_id", ""))
            if job.isdigit():
                if job in placements:
                    raise ValueError("duplicate job ID in receipt")
                placements[job] = row
            elif attempt["status"] not in {"submitting", "unknown"}:
                raise ValueError("invalid receipt job ID")
            attempts.append(attempt)
    if not collect:
        return {
            "dry_run": True,
            "manifest_sha256": digest,
            "attempts": len(attempts),
            "action": "--collect polls, copies completed runs, and uploads serially",
        }
    # ponytail: one local lock; retain a single collector instead of distributed locking.
    with (directory / "collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / "collection.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "manifest_sha256": digest,
                "collected": {},
                "upload_pending": False,
                "uploads": [],
            }
        )
        if (
            state.get("manifest_sha256") != digest
            or not set(state["collected"]) <= rows.keys()
        ):
            raise ValueError("collection state does not match the approved manifest")
        jobs = {}
        for alias in sorted({a["alias"] for a in attempts}):
            ids = [
                str(a["job_id"])
                for a in attempts
                if a["alias"] == alias and str(a.get("job_id", "")).isdigit()
            ]
            if ids:
                jobs.update(poll(alias, ids, runner))
        now = datetime.now(timezone.utc).isoformat()
        status = [
            {
                **a,
                **jobs.get(
                    str(a.get("job_id")),
                    {"state": "UNKNOWN", "live": False, "exit_code": None},
                ),
            }
            for a in attempts
        ]
        atomic_write_json(
            directory / "status.json",
            {"manifest_sha256": digest, "checked_at": now, "jobs": status},
        )
        check_receipts("polling")
        groups = {data: [a for a in attempts if a["data"] == data] for data in rows}
        ready = {data for data, group in groups.items() if eligible(group, jobs)}

        def destination(data):
            row = placements[state["collected"][data]]
            return directory / "production" / row["user"] / row["run_name"]

        for data, row in rows.items():
            group = groups[data]
            if data not in ready:
                continue
            job = str(group[-1]["job_id"])
            if state["collected"].get(data) == job:
                continue
            row = placements[job]
            target = directory / "production" / row["user"] / row["run_name"]
            target.mkdir(parents=True, exist_ok=True)
            runner(
                [
                    "rsync",
                    "-a",
                    "--timeout=60",
                    "-e",
                    shlex.join(SSH),
                    f"{row['alias']}:{row['run_dir']}/",
                    str(target) + "/",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            check_receipts("copy")
            state["collected"][data], state["upload_pending"] = job, True
            atomic_write_json(state_path, state)
        uploadable = [data for data in state["collected"] if data in ready]
        if state["upload_pending"] and uploadable:
            if uploader is None:
                from results.coordinator import upload_completed_runs

                uploader = upload_completed_runs
            try:
                check_receipts("collection")
                result = uploader(
                    [destination(data) for data in uploadable],
                    wandb_run_id=manifest["wandb"]["run_id"],
                    project=manifest["wandb"]["project"],
                    entity=manifest["wandb"]["entity"],
                )
            except Exception as error:
                state["uploads"].append({"at": now, "error": str(error)})
                atomic_write_json(state_path, state)
                raise
            state["uploads"].append({"at": now, "response": result})
            state["upload_pending"] = len(uploadable) != len(state["collected"])
        atomic_write_json(state_path, state)
        failures = [
            j
            for j in status
            if j["state"] in TERMINAL
            and (j["state"] != "COMPLETED" or j["exit_code"] != "0:0")
        ]
        latest = {a["data"]: a for a in attempts}
        return {
            "dry_run": False,
            "collected": len(state["collected"]),
            "failed_jobs": [
                j for j in failures if j["job_id"] == latest[j["data"]].get("job_id")
            ],
            "failed_attempts": failures,
            "last_upload": state["uploads"][-1] if state["uploads"] else None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--retry-approved-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.directory,
                approved_sha256=args.approved_sha256,
                retry_approved_sha256=args.retry_approved_sha256,
                collect=args.collect,
            ),
            indent=2,
        )
    )
