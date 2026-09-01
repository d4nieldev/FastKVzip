#!/usr/bin/env python3
"""Probe SLURM job state and push it to the dashboard server.

Runs inside the cluster as a CPU-only batch job and pushes outward over HTTPS,
so the dashboard needs no inbound access to BGU and no VPN.

Standard library only, on purpose: this must keep working even when the research
venv is mid-upgrade, so it never imports from ``$FASTKVZIP_VENV``.
"""

from __future__ import annotations

import argparse
import base64
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

AGENT_VERSION = 1

DEFAULT_POLL_SECONDS = 30
# sacct is the slow call; it only needs to catch jobs that aged out of scontrol.
SACCT_EVERY_N_POLLS = 10
SRES_EVERY_N_POLLS = 10

# Scripts collected per poll. Each costs a subprocess or two, and a first run
# against a month of sacct history would otherwise fire hundreds at once.
SCRIPTS_PER_POLL = 5

# How many times a still-live job may be re-asked before giving up. Where
# accounting keeps nothing, a running job's script is the only kind that can
# still be fetched, so one failed attempt must not lose it for good.
MAX_SCRIPT_ATTEMPTS = 20

# Tried in order; the first that prints anything wins. See collect_sres for why
# the interactive login shell has to come first.
SRES_ATTEMPTS = (
    ["bash", "-lic", "sres"],
    ["bash", "-lc", "sres"],
    ["bash", "-ic", "sres"],
)
# Per job, per poll. A backlog drains over successive polls instead of
# producing one enormous request.
MAX_LOG_CHUNK_BYTES = 512 * 1024
# How far back to start when a log is first discovered. Jobs are often already
# hours in with enormous tqdm-heavy logs; shipping those from byte 0 at the
# chunk cap would spend hours replaying old output before reaching what the job
# is doing now. The recent tail is what monitoring is for.
INITIAL_BACKFILL_BYTES = 2 * 1024 * 1024
# How far to look for a line ending when aligning the start of that window.
MAX_LINE_SCAN_BYTES = 64 * 1024
# Stop tailing a job's log this long after it finished.
TAIL_GRACE_SECONDS = 3600
# Bytes of a log's head hashed to detect it being rewritten in place.
HEAD_FINGERPRINT_BYTES = 4096
SACCT_LOOKBACK_DAYS = 30
RESUBMIT_MARGIN_SECONDS = 600
HTTP_TIMEOUT_SECONDS = 60

STATE_DIR = os.path.join(os.path.expanduser("~"), ".fastkvzip-dashboard")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
RESUBMIT_LOCK_PATH = os.path.join(STATE_DIR, "resubmit.lock")

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# SLURM value parsing
# --------------------------------------------------------------------------- #

# SLURM spells "no value" a dozen different ways.
_UNSET = {
    "",
    "unknown",
    "none",
    "(null)",
    "n/a",
    "unlimited",
    "partition_limit",
    # sacct's NodeList for a job that never got an allocation.
    "none assigned",
}

# States in which the job has actually been placed on a node. Anything else has
# not started, which matters because SLURM reports a *predicted* StartTime for
# pending jobs -- a future timestamp that must not be mistaken for a real one.
STARTED_STATES = {"RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT"}

_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+)-)?"
    r"(?:(?P<hours>\d+):)?"
    r"(?P<minutes>\d+):"
    r"(?P<seconds>\d+)(?:\.\d+)?$"
)


def clean(value: str | None) -> str | None:
    """Normalize a raw SLURM field, mapping its many null spellings to None."""
    if value is None:
        return None
    value = value.strip()
    return None if value.lower() in _UNSET else value


def parse_duration(value: str | None) -> int | None:
    """Parse ``D-HH:MM:SS`` / ``HH:MM:SS`` / ``MM:SS`` into seconds."""
    value = clean(value)
    if value is None:
        return None
    match = _DURATION_RE.match(value)
    if match is None:
        # sacct emits a bare integer for some elapsed fields.
        return int(value) if value.isdigit() else None
    parts = match.groupdict(default="0")
    return (
        int(parts["days"]) * 86400
        + int(parts["hours"]) * 3600
        + int(parts["minutes"]) * 60
        + int(parts["seconds"])
    )


def parse_timestamp(value: str | None) -> int | None:
    """Parse a SLURM ``YYYY-MM-DDTHH:MM:SS`` stamp into a UTC epoch int.

    SLURM prints local cluster time with no zone, so it is interpreted in the
    cluster's local timezone and normalized here. Everything downstream --
    database, API, browser -- deals only in UTC epochs.
    """
    value = clean(value)
    if value is None:
        return None
    try:
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return int(naive.astimezone().timestamp())


def parse_exit_code(value: str | None) -> str | None:
    """Keep ``0:0`` style codes, but drop them for jobs that simply succeeded."""
    value = clean(value)
    if value is None or value in {"0:0", "0"}:
        return None
    return value


def tres_field(tres: str | None, key: str) -> str | None:
    """Pull one ``key=value`` out of a SLURM TRES string."""
    if not tres:
        return None
    for item in tres.split(","):
        name, _, val = item.partition("=")
        if name.strip() == key:
            return val.strip() or None
    return None


def gres_from_tres(tres: str | None) -> str | None:
    """Extract the GPU spec, e.g. ``gres/gpu:rtx_pro_6000=1`` -> ``rtx_pro_6000:1``.

    SLURM lists both an untyped total and a per-type count
    (``gres/gpu=1,gres/gpu:rtx_pro_6000=1``); the typed entry is the useful one,
    so it wins regardless of which came first.
    """
    if not tres:
        return None
    untyped = None
    for item in tres.split(","):
        name, _, val = item.partition("=")
        name = name.strip()
        val = val.strip()
        if not name.startswith("gres/gpu"):
            continue
        if ":" in name:
            return f"{name.split(':', 1)[1]}:{val}"
        untyped = f"gpu:{val}"
    return untyped


def run_command_detail(argv: list[str], timeout: int = 60) -> tuple[str | None, str | None]:
    """Run a command, returning (stdout, why it failed).

    Exactly one side is ever set. The reason is kept rather than logged and
    dropped because the agent's log is on the cluster, which is the one place
    the dashboard exists to save you from having to look.

    Never raises: a cluster hiccup must degrade the poll, not kill the agent.
    """
    if shutil.which(argv[0]) is None:
        return None, f"{argv[0]} is not on PATH"
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # An interactive shell would otherwise inherit the batch job's
            # stdin and could sit waiting on it until the timeout.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if result.returncode != 0:
        stderr = [line for line in result.stderr.strip().splitlines() if line.strip()]
        return None, f"exit {result.returncode}: {stderr[-1][:200] if stderr else 'no stderr'}"
    return result.stdout, None


def run_command(argv: list[str], timeout: int = 60) -> str | None:
    """Run a SLURM command, returning stdout or None if it is unusable."""
    output, failure = run_command_detail(argv, timeout)
    if failure is not None and not failure.endswith("is not on PATH"):
        log(f"{argv[0]} failed: {failure}")
    return output


# --------------------------------------------------------------------------- #
# Source 1: scontrol -- live jobs (pending, running, recently finished)
# --------------------------------------------------------------------------- #

# scontrol's one-line format is space separated key=value, but values may
# themselves contain spaces (Reason, Command). Splitting on "key=" boundaries
# rather than on whitespace keeps those intact.
_KV_BOUNDARY_RE = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:\[\]-]*)=")


def parse_scontrol_line(line: str) -> dict[str, str]:
    matches = list(_KV_BOUNDARY_RE.finditer(line))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[match.group(1)] = line[match.end() : end].strip()
    return fields


def collect_scontrol_jobs(user: str) -> list[dict]:
    output = run_command(["scontrol", "show", "job", "-o"])
    if output is None:
        return []

    jobs = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("JobId="):
            continue
        fields = parse_scontrol_line(line)

        # UserId looks like "danieloh(1234)".
        owner = (fields.get("UserId") or "").split("(", 1)[0]
        if owner != user:
            continue

        # Array tasks report their own ArrayJobId/ArrayTaskId; prefer the
        # human-facing "123_4" id so it matches sacct and squeue.
        job_id = fields.get("ArrayJobId") and fields.get("ArrayTaskId")
        job_id = (
            f"{fields['ArrayJobId']}_{fields['ArrayTaskId']}"
            if job_id
            else fields.get("JobId", "")
        )
        if not job_id:
            continue

        state = (clean(fields.get("JobState")) or "UNKNOWN").upper()
        time_limit_s = parse_duration(fields.get("TimeLimit"))
        elapsed_s = parse_duration(fields.get("RunTime"))
        req_tres = clean(fields.get("ReqTRES"))
        alloc_tres = clean(fields.get("AllocTRES"))

        remaining_s = None
        if state == "RUNNING" and time_limit_s is not None and elapsed_s is not None:
            remaining_s = max(0, time_limit_s - elapsed_s)

        # For a queued job StartTime is SLURM's estimate of when it *will* run.
        # Reported as a real start it would put the job in the future, hiding it
        # from time-window queries and giving it a wall clock it has not begun.
        started = state in STARTED_STATES or state in TERMINAL_STATES
        start_time = parse_timestamp(fields.get("StartTime"))

        jobs.append(
            {
                "job_id": job_id,
                "name": clean(fields.get("JobName")),
                "user": owner,
                "state": state,
                "partition": clean(fields.get("Partition")),
                "reason": clean(fields.get("Reason")),
                "dependency": clean(fields.get("Dependency")),
                "exit_code": parse_exit_code(fields.get("ExitCode")),
                "submit_ts": parse_timestamp(fields.get("SubmitTime")),
                "start_ts": start_time if started else None,
                "est_start_ts": None if started else start_time,
                "end_ts": (
                    parse_timestamp(fields.get("EndTime"))
                    if state in TERMINAL_STATES
                    else None
                ),
                "elapsed_s": elapsed_s,
                "time_limit_s": time_limit_s,
                "remaining_s": remaining_s,
                "cpus": clean(fields.get("NumCPUs")),
                "nodes": clean(fields.get("NumNodes")),
                "node_list": clean(fields.get("NodeList")),
                "req_tres": req_tres,
                "alloc_tres": alloc_tres,
                "gres": gres_from_tres(alloc_tres) or gres_from_tres(req_tres),
                "mem_req": tres_field(req_tres, "mem"),
                "work_dir": clean(fields.get("WorkDir")),
                "command": clean(fields.get("Command")),
                "std_out": clean(fields.get("StdOut")),
                "source": "scontrol",
            }
        )
    return jobs


def collect_squeue_jobs(user: str) -> list[dict]:
    """Fallback for clusters where ``scontrol show job`` is restricted.

    Loses StdOut and TRES detail, but keeps state and wall-time accounting --
    which is the part that must never go dark.
    """
    fmt = "%i|%j|%T|%P|%V|%S|%M|%l|%L|%C|%D|%m|%b|%N|%R|%Z"
    output = run_command(["squeue", "-h", "-u", user, "-o", fmt])
    if output is None:
        return []

    jobs = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 16:
            continue
        state = (clean(parts[2]) or "UNKNOWN").upper()
        started = state in STARTED_STATES or state in TERMINAL_STATES
        start_time = parse_timestamp(parts[5])
        jobs.append(
            {
                "job_id": parts[0].strip(),
                "name": clean(parts[1]),
                "user": user,
                "state": state,
                "partition": clean(parts[3]),
                "reason": clean(parts[14]),
                "dependency": None,
                "exit_code": None,
                "submit_ts": parse_timestamp(parts[4]),
                "start_ts": start_time if started else None,
                "est_start_ts": None if started else start_time,
                "end_ts": None,
                "elapsed_s": parse_duration(parts[6]),
                "time_limit_s": parse_duration(parts[7]),
                "remaining_s": parse_duration(parts[8]),
                "cpus": clean(parts[9]),
                "nodes": clean(parts[10]),
                "node_list": clean(parts[13]),
                "req_tres": None,
                "alloc_tres": None,
                "gres": clean(parts[12]),
                "mem_req": clean(parts[11]),
                "work_dir": clean(parts[15]),
                "std_out": None,
                "source": "squeue",
            }
        )
    return jobs


# --------------------------------------------------------------------------- #
# Source 2: sacct -- history
# --------------------------------------------------------------------------- #

SACCT_FIELDS = [
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "Timelimit",
    "ReqTRES",
    "AllocTRES",
    "MaxRSS",
    "Partition",
    "NCPUS",
    "NNodes",
    "NodeList",
    "WorkDir",
]


def collect_sacct_jobs(user: str, lookback_days: int) -> list[dict]:
    start = f"now-{lookback_days}days"
    output = run_command(
        [
            "sacct",
            "-u",
            user,
            "-S",
            start,
            "-P",
            "-n",
            "--format=" + ",".join(SACCT_FIELDS),
        ],
        timeout=120,
    )
    if output is None:
        return []

    jobs: dict[str, dict] = {}
    # .batch / .extern steps carry MaxRSS but nothing else worth showing, so
    # they are folded into their parent rather than listed as jobs.
    step_rss: dict[str, str] = {}

    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < len(SACCT_FIELDS):
            continue
        row = dict(zip(SACCT_FIELDS, (part.strip() for part in parts)))
        raw_id = row["JobID"]

        if "." in raw_id:
            parent = raw_id.split(".", 1)[0]
            max_rss = clean(row["MaxRSS"])
            if max_rss:
                step_rss[parent] = max_rss
            continue

        # "CANCELLED by 1234" -> "CANCELLED"
        state = (clean(row["State"]) or "UNKNOWN").upper().split(" ")[0]
        req_tres = clean(row["ReqTRES"])
        alloc_tres = clean(row["AllocTRES"])
        time_limit_s = parse_duration(row["Timelimit"])
        elapsed_s = parse_duration(row["Elapsed"])
        started = state in STARTED_STATES or state in TERMINAL_STATES
        start_time = parse_timestamp(row["Start"])

        jobs[raw_id] = {
            "job_id": raw_id,
            "name": clean(row["JobName"]),
            "user": user,
            "state": state,
            "partition": clean(row["Partition"]),
            "reason": None,
            "dependency": None,
            "exit_code": parse_exit_code(row["ExitCode"]),
            "submit_ts": parse_timestamp(row["Submit"]),
            "start_ts": start_time if started else None,
            "est_start_ts": None if started else start_time,
            "end_ts": parse_timestamp(row["End"]) if state in TERMINAL_STATES else None,
            "elapsed_s": elapsed_s,
            "time_limit_s": time_limit_s,
            "remaining_s": (
                max(0, time_limit_s - elapsed_s)
                if state == "RUNNING"
                and time_limit_s is not None
                and elapsed_s is not None
                else None
            ),
            "cpus": clean(row["NCPUS"]),
            "nodes": clean(row["NNodes"]),
            "node_list": clean(row["NodeList"]),
            "req_tres": req_tres,
            "alloc_tres": alloc_tres,
            "gres": gres_from_tres(alloc_tres) or gres_from_tres(req_tres),
            "mem_req": tres_field(req_tres, "mem"),
            "max_rss": None,
            "work_dir": clean(row["WorkDir"]),
            "std_out": None,
            "source": "sacct",
        }

    for job_id, max_rss in step_rss.items():
        if job_id in jobs:
            jobs[job_id]["max_rss"] = max_rss
    return list(jobs.values())


def merge_jobs(live: list[dict], history: list[dict]) -> list[dict]:
    """Combine both sources, letting live data win field by field.

    scontrol is authoritative for anything it reports (it is current), but
    sacct supplies MaxRSS and fills in jobs scontrol has already forgotten.
    """
    merged: dict[str, dict] = {job["job_id"]: dict(job) for job in history}
    for job in live:
        existing = merged.get(job["job_id"])
        if existing is None:
            merged[job["job_id"]] = dict(job)
            continue
        for key, value in job.items():
            if value is not None:
                existing[key] = value
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Log tailing
# --------------------------------------------------------------------------- #


def resolve_log_path(job: dict, cached: str | None) -> str | None:
    """Find the job's log file, preferring what SLURM itself reported."""
    if job.get("std_out"):
        return job["std_out"]
    if cached:
        return cached
    # sacct does not report StdOut, so fall back to this repo's convention:
    # .slurm/logs/<jobid>-<runname>.log, relative to the submit directory.
    work_dir = job.get("work_dir")
    if not work_dir:
        return None
    matches = sorted(glob.glob(os.path.join(work_dir, ".slurm", "logs", f"{job['job_id']}-*.log")))
    return matches[0] if matches else None


def should_tail(job: dict, now: int) -> bool:
    if job["state"] not in TERMINAL_STATES:
        return True
    end_ts = job.get("end_ts")
    # No end time recorded: tail it once rather than lose the log entirely.
    return end_ts is None or (now - end_ts) < TAIL_GRACE_SECONDS


def head_fingerprint(path: str) -> str | None:
    """Hash the first bytes of a log, to notice it being rewritten in place.

    Shrink and inode checks miss a log replaced with *longer* content at the
    same path -- the agent would resume mid-file and stitch new output onto
    stale bytes. Comparing the head catches that.

    Returns None until the file is at least one window long, so that a log
    still growing through its first few kilobytes -- where the head legitimately
    changes on every append -- is not mistaken for a rewrite.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(HEAD_FINGERPRINT_BYTES)
    except OSError:
        return None
    if len(head) < HEAD_FINGERPRINT_BYTES:
        return None
    return hashlib.sha256(head).hexdigest()


def read_log_chunk(path: str, offset: int) -> tuple[bytes, int, bool] | None:
    """Read up to MAX_LOG_CHUNK_BYTES from ``offset``.

    Returns (data, effective_offset, rewound). ``rewound`` is True when the file
    shrank -- a rewritten log -- in which case reading restarts at byte 0.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None

    rewound = False
    if offset > size:
        offset = 0
        rewound = True
    if offset == size:
        return b"", offset, rewound

    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(MAX_LOG_CHUNK_BYTES)
    except OSError as exc:
        log(f"cannot read {path}: {exc}")
        return None
    return data, offset, rewound


def _start_from_tail(entry: dict, path: str) -> None:
    """Point a log entry at the last INITIAL_BACKFILL_BYTES of the file.

    ``base_offset`` is the cluster-file byte that the server stores as byte 0,
    so server offsets stay a small window even when the cluster file is huge.
    The window is nudged forward to the next newline, since cutting at an
    arbitrary byte would make the first line in the viewer a fragment.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0

    base = max(0, size - INITIAL_BACKFILL_BYTES)
    if base > 0:
        try:
            with open(path, "rb") as handle:
                handle.seek(base)
                partial = handle.readline(MAX_LINE_SCAN_BYTES)
            # Skip the fragment only when a line ends within reach AND doing so
            # leaves most of the window intact. A log whose lines are longer
            # than the window would otherwise skip past everything.
            if partial.endswith(b"\n") and len(partial) < INITIAL_BACKFILL_BYTES:
                base += len(partial)
        except OSError:
            pass

    entry["base_offset"] = base
    entry["file_pos"] = base
    entry["truncate_next"] = True


def build_log_payloads(
    jobs: list[dict], state: dict, now: int
) -> tuple[list[dict], dict[str, int]]:
    """Collect new log bytes, plus a declaration of where each log stands.

    Returns (chunks, offsets). The offsets cover every log being tailed, even
    those with nothing new, so the server can spot a log it no longer holds --
    a job whose output has gone quiet would otherwise never reveal the gap.

    Offsets sent to the server are relative to ``base_offset``, not to the
    cluster file, so a log first seen mid-run ships its tail rather than
    replaying hours of history at the per-poll chunk cap.
    """
    tracked = state.setdefault("logs", {})
    payloads = []
    offsets: dict[str, int] = {}

    for job in jobs:
        job_id = job["job_id"]
        entry = tracked.setdefault(
            job_id,
            {"path": None, "base_offset": 0, "file_pos": 0, "inode": None, "head": None},
        )

        path = resolve_log_path(job, entry.get("path"))
        if path is None:
            continue
        if path != entry.get("path"):
            entry["path"] = path
            entry["inode"] = None
            entry["head"] = None
            _start_from_tail(entry, path)
        if not should_tail(job, now):
            continue

        try:
            inode = os.stat(path).st_ino
        except OSError:
            continue
        # A new inode at the same path means the file was replaced, not appended.
        replaced = entry.get("inode") is not None and inode != entry["inode"]
        entry["inode"] = inode

        # Truncated and rewritten in place keeps the inode, so compare the head
        # too -- otherwise new output would be stitched onto stale bytes.
        head = head_fingerprint(path)
        if entry.get("head") is not None and head is not None and head != entry["head"]:
            replaced = True
        entry["head"] = head

        if replaced:
            # A replaced file is a new run's output: take it from the start.
            entry["base_offset"] = 0
            entry["file_pos"] = 0
            entry["truncate_next"] = True

        chunk = read_log_chunk(path, int(entry.get("file_pos") or 0))
        if chunk is None:
            continue
        data, file_pos, rewound = chunk
        if rewound:
            entry["base_offset"] = 0
            entry["file_pos"] = 0
            entry["truncate_next"] = True
            file_pos = 0

        base = int(entry.get("base_offset") or 0)
        offsets[job_id] = max(0, file_pos - base)
        if not data:
            continue

        payloads.append(
            {
                "job_id": job_id,
                "path": path,
                "offset": max(0, file_pos - base),
                # Replace rather than append: either the cluster file was
                # rewritten, or this is a fresh tail window that the server's
                # existing content does not line up with.
                "truncate": bool(entry.get("truncate_next")),
                "encoding": "gzip+base64",
                "data": base64.b64encode(gzip.compress(data)).decode("ascii"),
                "raw_bytes": len(data),
            }
        )
    return payloads, offsets


# --------------------------------------------------------------------------- #
# Local state
# --------------------------------------------------------------------------- #


def load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {"logs": {}}
    if not isinstance(state, dict):
        return {"logs": {}}
    state.setdefault("logs", {})
    return state


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp_path = f"{STATE_PATH}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp_path, STATE_PATH)
    except OSError as exc:
        log(f"cannot persist state: {exc}")


def prune_state(state: dict, known_ids: set[str]) -> None:
    """Forget jobs the cluster no longer reports, so state.json stays bounded."""
    for bucket in ("logs", "scripts"):
        entries = state.get(bucket, {})
        for job_id in [key for key in entries if key not in known_ids]:
            del entries[job_id]


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def post_payload(url: str, token: str, payload: dict) -> dict | None:
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    request = urllib.request.Request(
        url.rstrip("/") + "/api/ingest",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "X-Agent-Token": token,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        log(f"ingest rejected ({exc.code}): {detail}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"ingest failed: {exc}")
    return None


def apply_server_response(response: dict, state: dict) -> None:
    """Let the server drive our offsets.

    The server is the authority on how much log it holds. If its disk was wiped
    -- likely on a free tier with ephemeral storage -- it answers with a reset
    and we re-ship that log from byte 0 on the next poll.
    """
    logs = state.setdefault("logs", {})
    for job_id, next_offset in (response.get("ack") or {}).items():
        entry = logs.get(job_id)
        if entry is None:
            continue
        # The ack is a server offset; translate it back to a cluster position.
        entry["file_pos"] = int(entry.get("base_offset") or 0) + int(next_offset)
        entry["truncate_next"] = False

    for job_id in response.get("reset") or []:
        entry = logs.get(job_id)
        if entry is None or not entry.get("path"):
            continue
        # Re-ship a fresh tail rather than everything since base_offset, so a
        # wiped server costs one window per job instead of the whole backlog.
        _start_from_tail(entry, entry["path"])
        log(f"server requested log reset for job {job_id}")


# --------------------------------------------------------------------------- #
# Self-resubmission
# --------------------------------------------------------------------------- #


def job_end_time() -> int | None:
    """Epoch at which this agent's own allocation expires."""
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if raw and raw.isdigit():
        return int(raw)

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    output = run_command(["scontrol", "show", "job", "-o", job_id])
    if not output:
        return None
    return parse_timestamp(parse_scontrol_line(output.splitlines()[0]).get("EndTime"))


def resubmit_self(script_path: str) -> bool:
    """Queue a successor agent chained after this one, exactly once."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        log("not running under SLURM; skipping resubmission")
        return False
    if os.path.exists(RESUBMIT_LOCK_PATH):
        try:
            with open(RESUBMIT_LOCK_PATH, "r", encoding="utf-8") as handle:
                marker = handle.read().strip()
        except OSError:
            marker = ""
        # The lock is per predecessor, so a stale one from an older agent
        # must not block this generation from continuing the chain.
        if marker.startswith(f"{job_id}:"):
            log(f"successor already queued ({marker})")
            return True

    output = run_command(
        ["sbatch", "--parsable", f"--dependency=afterany:{job_id}", script_path]
    )
    if output is None:
        log("resubmission failed; the dashboard will go stale when this job ends")
        return False

    successor = output.strip().split(";")[0]
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        with open(RESUBMIT_LOCK_PATH, "w", encoding="utf-8") as handle:
            handle.write(f"{job_id}:{successor}")
    except OSError:
        pass
    log(f"queued successor agent job {successor}")
    return True


# --------------------------------------------------------------------------- #
# Poll
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Source 4: the submitted script and the environment it came from
# --------------------------------------------------------------------------- #

MAX_SCRIPT_BYTES = 256 * 1024
MAX_ENV_BYTES = 64 * 1024


_ACCOUNTING_FLAGS: str | None = None


def accounting_flags() -> str:
    """This cluster's AccountingStoreFlags, read once.

    It decides what can honestly be said when a script is missing. With the
    flag unset there is nothing vague about it: the script is gone and no
    amount of retrying will bring it back, which is worth saying plainly
    rather than hedging about site configuration.
    """
    global _ACCOUNTING_FLAGS
    if _ACCOUNTING_FLAGS is None:
        output = run_command(["scontrol", "show", "config"], timeout=30) or ""
        _ACCOUNTING_FLAGS = ""
        for line in output.splitlines():
            if line.strip().lower().startswith("accountingstoreflags"):
                _, _, value = line.partition("=")
                value = value.strip()
                _ACCOUNTING_FLAGS = "" if value in ("(null)", "") else value
                break
    return _ACCOUNTING_FLAGS


def collect_batch_script(job_id: str, command: str | None) -> tuple[str | None, str]:
    """The script this job was submitted with, from whichever source still has it.

    Three sources, in descending order of how much they can be trusted:

    * ``scontrol write batch_script`` reads slurmctld's own copy, which is
      exactly what was submitted -- but the controller forgets a job MinJobAge
      seconds after it ends (five minutes by default).
    * ``sacct --batch-script`` reads the same bytes back out of the accounting
      database, and keeps them for as long as accounting does. It only answers
      when the site set ``AccountingStoreFlags=job_script``, which is off by
      default, so it may simply not be available here.
    * The file named by the job's Command, as it stands on disk *now*. Not the
      same thing -- the repository may have moved on since submission -- so it
      is reported under its own source name and never passed off as the other
      two.
    """
    # scontrol writes to a file, not to stdout, so it needs somewhere to write.
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "batch_script")
        _, failure = run_command_detail(
            ["scontrol", "write", "batch_script", job_id, target], timeout=30
        )
        if failure is None:
            body = read_text_file(target, MAX_SCRIPT_BYTES)
            if body:
                return body, "scontrol"

    body, failure = run_command_detail(["sacct", "-j", job_id, "--batch-script"], timeout=30)
    if failure is None and body and body.strip():
        return body.strip()[:MAX_SCRIPT_BYTES], "sacct"

    path = clean(command)
    if path and os.path.isabs(path) and os.path.isfile(path):
        body = read_text_file(path, MAX_SCRIPT_BYTES)
        if body:
            return body, "disk"

    return None, "unavailable"


def collect_job_env(job_id: str) -> tuple[str | None, str]:
    """The environment the job was submitted from.

    Only accounting keeps this, and only when the site set
    ``AccountingStoreFlags=job_env``. There is no second source: the
    environment a finished job ran under exists nowhere else on the cluster.
    """
    body, failure = run_command_detail(["sacct", "-j", job_id, "--env-vars"], timeout=30)
    if failure is None and body and body.strip():
        return body.strip()[:MAX_ENV_BYTES], "sacct"
    return None, "unavailable"


def read_text_file(path: str, limit: int) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            body = handle.read(limit)
    except OSError:
        return None
    return body or None


def missing_script_note(terminal: bool) -> str:
    """Why this job has no script, in terms of what this cluster actually does."""
    flags = accounting_flags()
    if "job_script" in flags:
        return (
            "SLURM no longer has this job's script: the controller has "
            "forgotten the job and accounting did not return it either."
        )
    return (
        "This cluster keeps no job scripts in accounting "
        "(AccountingStoreFlags is unset), so a script can only be taken from "
        "the controller while the job is still queued or running. "
        + (
            "This job had already finished when the dashboard first saw it."
            if terminal
            else "The dashboard will keep trying while this job is alive."
        )
    )


def build_script_payloads(jobs: list[dict], state: dict) -> list[dict]:
    """Collect scripts for jobs that do not have one yet.

    Live jobs are asked about first, and asked again until they answer. Where
    accounting stores nothing, the controller is the only source and it forgets
    a job MinJobAge after it ends, so a running job's script is the one that is
    still there to be had -- and a backlog of finished jobs from sacct must not
    spend the per-poll budget ahead of it. A finished job is asked once, since
    it either ended within the last few minutes or it is already too late.
    """
    tracked = state.setdefault("scripts", {})
    ordered = sorted(jobs, key=lambda job: job.get("state") in TERMINAL_STATES)

    payloads = []
    for job in ordered:
        if len(payloads) >= SCRIPTS_PER_POLL:
            break
        job_id = job["job_id"]
        record = tracked.get(job_id) or {}
        if record.get("done") or record.get("tries", 0) >= MAX_SCRIPT_ATTEMPTS:
            continue

        terminal = job.get("state") in TERMINAL_STATES
        script, script_source = collect_batch_script(job_id, job.get("command"))
        env, env_source = collect_job_env(job_id)

        note = None
        if script is None:
            note = missing_script_note(terminal)
        if env is None and "job_env" not in accounting_flags():
            env_note = (
                "The submission environment is stored only where the site sets "
                "AccountingStoreFlags=job_env, which this cluster does not."
            )
            note = f"{note} {env_note}" if note else env_note

        tracked[job_id] = {
            "tries": record.get("tries", 0) + 1,
            # Settled once the script is in hand, or once the job is over and
            # the controller has no more to give. A live job that failed stays
            # open, because next poll it may still answer.
            "done": script is not None or terminal,
        }
        payloads.append(
            {
                "job_id": job_id,
                "batch_script": script,
                "job_env": env,
                "script_source": script_source,
                "env_source": env_source,
                "note": note,
            }
        )
    return payloads


def collect_sres() -> str | None:
    """Snapshot BGU's site-local GPU availability command.

    `sres` is not a real executable -- exec'ing it gives "Exec format error" --
    so it is an alias, a shell function, or an unheadered script, and only a
    shell can run it. *Which* shell turns out to matter: a non-interactive bash
    leaves `expand_aliases` off, so an alias never resolves, and `bash -l`
    sources ~/.bash_profile but never ~/.bashrc, so a function defined there is
    invisible to `bash -lc`. The interactive login shell is tried first for
    that reason, the plainer forms after it.

    A failure is described rather than dropped. Returning None stores nothing,
    the panel then renders nothing, and a command that has never once worked
    looks identical to one the agent has simply not got round to yet.
    """
    failures = []
    for argv in SRES_ATTEMPTS:
        output, failure = run_command_detail(argv, timeout=30)
        if output and output.strip():
            return output.strip()
        failures.append(f"    {' '.join(argv)}  ->  {failure or 'ran, but printed nothing'}")
    return (
        f"sres produced no output on {socket.gethostname()}.\n\n"
        + "\n".join(failures)
        + "\n\nIf sres only exists on the login node this is expected: the agent\n"
        "runs wherever SLURM placed it. Check with `type sres` on both."
    )


def build_payload(
    user: str,
    state: dict,
    *,
    include_history: bool,
    include_sres: bool,
    lookback_days: int,
    poll_seconds: int,
) -> dict:
    now = int(time.time())

    live = collect_scontrol_jobs(user)
    if not live:
        live = collect_squeue_jobs(user)
    history = collect_sacct_jobs(user, lookback_days) if include_history else []
    jobs = merge_jobs(live, history)

    self_job_id = os.environ.get("SLURM_JOB_ID")
    for job in jobs:
        job["is_agent"] = bool(self_job_id) and job["job_id"] == self_job_id

    prune_state(state, {job["job_id"] for job in jobs})
    logs, log_offsets = build_log_payloads(jobs, state, now)
    scripts = build_script_payloads(jobs, state)

    payload = {
        "agent": {
            "version": AGENT_VERSION,
            "job_id": self_job_id,
            "host": socket.gethostname(),
            "user": user,
            "cluster_time": now,
            "poll_interval": poll_seconds,
        },
        "jobs": jobs,
        "logs": logs,
        "log_offsets": log_offsets,
        "scripts": scripts,
        "full_refresh": include_history,
    }
    if include_sres:
        payload["sres"] = collect_sres()
    return payload


def poll_once(args: argparse.Namespace, state: dict, tick: int) -> None:
    payload = build_payload(
        args.user,
        state,
        include_history=(tick % args.sacct_every == 0),
        include_sres=(tick % SRES_EVERY_N_POLLS == 0),
        lookback_days=args.lookback_days,
        poll_seconds=args.interval,
    )

    if args.dry_run:
        # Logs are re-read next poll anyway; print sizes, not megabytes of text.
        preview = dict(payload)
        preview["logs"] = [
            {key: value for key, value in entry.items() if key != "data"}
            for entry in payload["logs"]
        ]
        json.dump(preview, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    response = post_payload(args.url, args.token, payload)
    if response is None:
        # Offsets are not advanced, so this poll's bytes are re-sent next time.
        return

    apply_server_response(response, state)
    save_state(state)

    shipped = sum(entry["raw_bytes"] for entry in payload["logs"])
    log(
        f"pushed {len(payload['jobs'])} jobs, "
        f"{len(payload['logs'])} log chunks ({shipped} bytes)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("DASHBOARD_URL", ""),
        help="dashboard base URL (env: DASHBOARD_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DASHBOARD_TOKEN", ""),
        help="ingest token (env: DASHBOARD_TOKEN)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("DASHBOARD_USER") or os.environ.get("USER") or "",
        help="cluster username to report on",
    )
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--sacct-every", type=int, default=SACCT_EVERY_N_POLLS)
    parser.add_argument("--lookback-days", type=int, default=SACCT_LOOKBACK_DAYS)
    parser.add_argument(
        "--script-path",
        default=os.environ.get("DASHBOARD_AGENT_SCRIPT", ""),
        help="sbatch script to resubmit as this agent's successor",
    )
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payload instead of sending it; no network, no state writes",
    )
    args = parser.parse_args(argv)

    if not args.user:
        parser.error("no username: pass --user or set USER")
    if not args.dry_run and not args.url:
        parser.error("no server: pass --url or set DASHBOARD_URL")

    state = load_state()

    if args.once:
        poll_once(args, state, tick=0)
        return 0

    end_time = job_end_time()
    if end_time:
        log(f"allocation ends at {datetime.fromtimestamp(end_time)}")

    log(f"polling every {args.interval}s for user {args.user} -> {args.url}")
    tick = 0
    while True:
        try:
            poll_once(args, state, tick)
        except Exception as exc:  # noqa: BLE001 - a bad poll must not end the agent
            log(f"poll failed: {exc}")

        if end_time and args.script_path:
            if end_time - time.time() < RESUBMIT_MARGIN_SECONDS:
                if resubmit_self(args.script_path):
                    log("wall time nearly exhausted; handing over to successor")
                    return 0
                # Could not queue a successor -- a submit limit, a busy
                # controller. Keep polling and retry on the next tick: the
                # remaining minutes of this allocation are still useful, and
                # giving up here would end the chain for good.

        tick += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
