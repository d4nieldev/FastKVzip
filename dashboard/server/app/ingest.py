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
    placeholders = ", ".join(f"%({column})s" for column in columns)
    # Every generated identifier is quoted, because one of them is "user",
    # which Postgres reads as the session user when left bare.
    names = ", ".join(f'"{column}"' for column in columns)
    # first_seen and seen_at are deliberately excluded from the update: the
    # former is a discovery timestamp, the latter the user's own reading
    # history, and neither is the agent's to overwrite on every poll.
    # is_agent is each agent's answer to "is this job me", so the successor
    # after a handover reports the job it replaced as *not* the agent. Keeping
    # it sticky is what stops a retired agent job rejoining the list the moment
    # its replacement starts.
    updates = ", ".join(
        "is_agent = GREATEST(jobs.is_agent, excluded.is_agent)"
        if column == "is_agent"
        else f'"{column}" = excluded."{column}"'
        for column in db.JOB_COLUMNS
    )

    connection.cursor().executemany(
        f"INSERT INTO jobs ({names}) VALUES ({placeholders}) "
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
            "VALUES (%s, %s, %s, %s) "
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


UNKNOWN_USER = "unknown"


def record_heartbeat(connection, agent: dict, now: int) -> None:
    """Record one agent's liveness, under the user it reports for.

    The user is the identity here, so an agent that names nobody is filed under
    a placeholder rather than silently taking over another agent's row.
    """
    user = _text(agent.get("user"), 64) or UNKNOWN_USER
    connection.execute(
        'INSERT INTO agents ("user", last_heartbeat, job_id, host, version, '
        "poll_interval, cluster_time) VALUES (%s, %s, %s, %s, %s, %s, %s) "
        'ON CONFLICT ("user") DO UPDATE SET '
        "last_heartbeat = excluded.last_heartbeat, job_id = excluded.job_id, "
        "host = excluded.host, version = excluded.version, "
        "poll_interval = excluded.poll_interval, cluster_time = excluded.cluster_time",
        (
            user,
            now,
            _text(agent.get("job_id"), 64),
            _text(agent.get("host"), 256),
            _int(agent.get("version")),
            _int(agent.get("poll_interval")),
            _int(agent.get("cluster_time")),
        ),
    )


def record_sres(connection, body: str | None, now: int) -> None:
    if body is None:
        return
    connection.execute(
        "INSERT INTO sres_snapshot (id, body, updated_at) VALUES (1, %s, %s) "
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
        # Only when the payload actually carries an agent block. A bare
        # {"jobs": [...]} -- which the tests and any manual backfill send --
        # should not conjure an agent into the user list.
        if agent:
            record_heartbeat(connection, agent, now)
        if "sres" in payload:
            record_sres(connection, payload.get("sres"), now)

        # Nothing but the agent's own job on file. Either this server is new or
        # its disk was wiped, and the agent's history sweep is only every tenth
        # poll -- so ask for it now rather than showing an empty dashboard for
        # the next five minutes. Logs already recover this way; jobs did not.
        needs_history = not payload.get("full_refresh") and not connection.execute(
            "SELECT 1 FROM jobs WHERE is_agent = 0 LIMIT 1"
        ).fetchone()

    return {
        "ack": ack,
        "reset": reset,
        "jobs_accepted": job_count,
        "server_time": now,
        "want_history": bool(needs_history),
    }
