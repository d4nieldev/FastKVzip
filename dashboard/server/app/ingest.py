"""Apply an agent payload to the database and log store."""

from __future__ import annotations

import base64
import binascii
import gzip
import time

from . import db, logstore

MAX_STATE_LENGTH = 32
MAX_TEXT_LENGTH = 2048


def _text(value, limit: int = MAX_TEXT_LENGTH) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _body(value, limit: int) -> str | None:
    """A long text field, kept verbatim.

    Unlike _text this does not strip: in a shell script the leading shebang
    line and the trailing newline are content, not padding.
    """
    if not isinstance(value, str) or not value:
        return None
    return value[:limit]


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_job(raw: dict, now: int) -> dict | None:
    """Coerce one agent-supplied job into the exact shape of the jobs table."""
    job_id = _text(raw.get("job_id"), 64)
    if not job_id:
        return None
    try:
        logstore.log_path(job_id)
    except logstore.UnsafeJobId:
        return None

    job = {"job_id": job_id}
    for column in db.JOB_COLUMNS:
        value = raw.get(column)
        if column == "is_agent":
            job[column] = 1 if value else 0
        elif column.endswith("_ts") or column.endswith("_s"):
            job[column] = _int(value)
        elif column == "state":
            job[column] = (_text(value, MAX_STATE_LENGTH) or "UNKNOWN").upper()
        else:
            job[column] = _text(value)
    job["last_seen"] = now
    job["first_seen"] = now
    return job


def upsert_jobs(connection, jobs: list[dict], now: int) -> int:
    rows = [job for job in (normalize_job(raw, now) for raw in jobs) if job]
    if not rows:
        return 0

    columns = ["job_id", *db.JOB_COLUMNS, "first_seen", "last_seen"]
    placeholders = ", ".join(f":{column}" for column in columns)
    # first_seen and seen_at are deliberately excluded from the update: the
    # former is a discovery timestamp, the latter the user's own reading
    # history, and neither is the agent's to overwrite on every poll.
    # is_agent is each agent's answer to "is this job me", so the successor
    # after a handover reports the job it replaced as *not* the agent. Keeping
    # it sticky is what stops a retired agent job rejoining the list the moment
    # its replacement starts.
    updates = ", ".join(
        "is_agent = MAX(jobs.is_agent, excluded.is_agent)"
        if column == "is_agent"
        else f"{column} = excluded.{column}"
        for column in db.JOB_COLUMNS
    )

    connection.executemany(
        f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(job_id) DO UPDATE SET {updates}, last_seen = excluded.last_seen",
        rows,
    )
    return len(rows)


def decode_chunk(entry: dict) -> bytes | None:
    data = entry.get("data")
    if not isinstance(data, str):
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if entry.get("encoding") == "gzip+base64":
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError, gzip.BadGzipFile):
            return None
    return raw


def apply_logs(connection, entries: list[dict], now: int) -> tuple[dict[str, int], list[str]]:
    """Append log chunks; return per-job accepted offsets and jobs needing reset."""
    ack: dict[str, int] = {}
    reset: list[str] = []

    for entry in entries:
        job_id = _text(entry.get("job_id"), 64)
        if not job_id:
            continue
        offset = _int(entry.get("offset"))
        data = decode_chunk(entry)
        if offset is None or offset < 0 or data is None:
            reset.append(job_id)
            continue

        try:
            next_offset, accepted = logstore.append(
                job_id, offset, data, truncate=bool(entry.get("truncate"))
            )
        except logstore.UnsafeJobId:
            continue

        ack[job_id] = next_offset
        if not accepted:
            reset.append(job_id)
            continue

        connection.execute(
            "INSERT INTO job_logs (job_id, path, size_bytes, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET "
            "path = excluded.path, size_bytes = excluded.size_bytes, "
            "updated_at = excluded.updated_at",
            (job_id, _text(entry.get("path")), next_offset, now),
        )
    return ack, reset


def verify_offsets(claimed: dict, ack: dict[str, int], reset: list[str]) -> None:
    """Reconcile the agent's offsets against what is actually stored.

    Log chunks alone are not enough to detect a lost server-side log: a job
    whose output has gone quiet ships no chunks, so nothing would ever reveal
    the gap and the log would stay missing. The agent therefore declares where
    it believes each tailed log stands, and anything the store cannot back up
    is answered with a reset.
    """
    for job_id, raw_offset in (claimed or {}).items():
        job_id = _text(job_id, 64)
        offset = _int(raw_offset)
        if not job_id or offset is None or job_id in ack:
            continue
        try:
            size = logstore.current_size(job_id)
        except logstore.UnsafeJobId:
            continue
        if offset > size:
            # We hold less than the agent has sent -- a wiped or pruned disk.
            ack[job_id] = size
            reset.append(job_id)


def record_heartbeat(connection, agent: dict, now: int) -> None:
    connection.execute(
        "INSERT INTO agent_status (id, last_heartbeat, job_id, host, user, version, "
        "poll_interval, cluster_time) VALUES (1, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "last_heartbeat = excluded.last_heartbeat, job_id = excluded.job_id, "
        "host = excluded.host, user = excluded.user, version = excluded.version, "
        "poll_interval = excluded.poll_interval, cluster_time = excluded.cluster_time",
        (
            now,
            _text(agent.get("job_id"), 64),
            _text(agent.get("host"), 256),
            _text(agent.get("user"), 64),
            _int(agent.get("version")),
            _int(agent.get("poll_interval")),
            _int(agent.get("cluster_time")),
        ),
    )


MAX_SCRIPT_LENGTH = 256 * 1024
MAX_ENV_LENGTH = 64 * 1024


def record_scripts(connection, entries: list[dict], now: int) -> int:
    """Store submitted scripts and environments the agent managed to collect.

    Written once and left alone: neither can change after submission, and the
    agent only sends an entry the first time it gets one.
    """
    rows = []
    for entry in entries:
        job_id = _text(entry.get("job_id"), 64)
        if not job_id:
            continue
        rows.append(
            {
                "job_id": job_id,
                "batch_script": _body(entry.get("batch_script"), MAX_SCRIPT_LENGTH),
                "job_env": _body(entry.get("job_env"), MAX_ENV_LENGTH),
                "script_source": _text(entry.get("script_source"), 64),
                "env_source": _text(entry.get("env_source"), 64),
                "note": _text(entry.get("note"), MAX_TEXT_LENGTH),
                "updated_at": now,
            }
        )
    if not rows:
        return 0
    connection.executemany(
        "INSERT INTO job_scripts "
        "(job_id, batch_script, job_env, script_source, env_source, note, updated_at) "
        "VALUES (:job_id, :batch_script, :job_env, :script_source, :env_source, "
        ":note, :updated_at) "
        "ON CONFLICT(job_id) DO UPDATE SET "
        "batch_script = COALESCE(excluded.batch_script, job_scripts.batch_script), "
        "job_env = COALESCE(excluded.job_env, job_scripts.job_env), "
        "script_source = COALESCE(excluded.script_source, job_scripts.script_source), "
        "env_source = COALESCE(excluded.env_source, job_scripts.env_source), "
        "note = excluded.note, updated_at = excluded.updated_at",
        rows,
    )
    return len(rows)


def record_sres(connection, body: str | None, now: int) -> None:
    if body is None:
        return
    connection.execute(
        "INSERT INTO sres_snapshot (id, body, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET body = excluded.body, updated_at = excluded.updated_at",
        (str(body)[:20000], now),
    )


def apply_payload(payload: dict) -> dict:
    """Apply a full agent payload; returns the ack/reset response body."""
    now = int(time.time())
    agent = payload.get("agent") or {}
    jobs = payload.get("jobs") or []
    logs = payload.get("logs") or []

    with db.transaction() as connection:
        job_count = upsert_jobs(connection, jobs, now)
        ack, reset = apply_logs(connection, logs, now)
        verify_offsets(payload.get("log_offsets"), ack, reset)
        record_scripts(connection, payload.get("scripts") or [], now)
        record_heartbeat(connection, agent, now)
        if "sres" in payload:
            record_sres(connection, payload.get("sres"), now)

    return {"ack": ack, "reset": reset, "jobs_accepted": job_count, "server_time": now}
