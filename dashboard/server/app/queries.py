"""Read queries and retention for the dashboard API."""

from __future__ import annotations

import os
import time

from . import db, logstore

RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "30"))

# States SLURM considers final. Anything else is treated as in-flight, which
# matters for the window query: a running job has no end time yet.
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

FAILURE_STATES = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "CANCELLED",
}


def _row_to_job(row) -> dict:
    job = dict(row)
    job["is_agent"] = bool(job["is_agent"])
    job["is_terminal"] = job["state"] in TERMINAL_STATES
    job["is_failure"] = job["state"] in FAILURE_STATES
    # A finished run the user has not looked at *since it finished*. Comparing
    # against end_ts rather than testing for null is what makes a job you
    # watched running still announce how it turned out.
    seen_at = job.get("seen_at")
    job["unseen"] = bool(
        job["is_terminal"] and (seen_at is None or (job["end_ts"] and seen_at < job["end_ts"]))
    )
    return job


def list_jobs(
    *,
    window_from: int | None = None,
    window_to: int | None = None,
    states: list[str] | None = None,
    search: str | None = None,
    include_agent: bool = False,
) -> list[dict]:
    """Jobs overlapping the given time window.

    Overlap, not containment: a job counts if it was alive at any point inside
    the window, which is what "still running or stopped in this window" means.
    A job with no end time is treated as running up to now.
    """
    now = int(time.time())
    clauses: list[str] = []
    params: list = []

    if window_from is not None:
        # Job ended (or is still going) at or after the window start.
        clauses.append("COALESCE(j.end_ts, ?) >= ?")
        params.extend([now, window_from])
    if window_to is not None:
        # Job began (or was queued) at or before the window end.
        clauses.append("COALESCE(j.start_ts, j.submit_ts, j.first_seen) <= ?")
        params.append(window_to)
    if states:
        clauses.append(f"j.state IN ({', '.join('?' for _ in states)})")
        params.extend(state.upper() for state in states)
    if search:
        clauses.append("(j.name LIKE ? OR j.job_id LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])
    if not include_agent:
        # The dashboard's own job is infrastructure, not an experiment. Its
        # health is the banner at the top of the page, which is where it can
        # still be opened from; in the list it is one more row to read past.
        clauses.append("j.is_agent = 0")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        # log_path is included here as well as in get_job: the detail panel
        # renders straight from the list response rather than refetching.
        "SELECT j.*, COALESCE(l.size_bytes, 0) AS log_bytes, l.path AS log_path "
        "FROM jobs j LEFT JOIN job_logs l ON l.job_id = j.job_id "
        f"{where} "
        # Newest job id first. The CAST is what makes it numeric: compared as
        # text, "9" would outrank "1000". Array tasks all share a base id, so
        # the part after the underscore breaks that tie; SUBSTR returns the
        # whole id when there is no underscore, which is harmless because a
        # plain id never ties with another.
        "ORDER BY CAST(j.job_id AS INTEGER) DESC, "
        "CAST(SUBSTR(j.job_id, INSTR(j.job_id, '_') + 1) AS INTEGER) DESC"
    )

    with db.connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_row_to_job(row) for row in rows]


def get_job(job_id: str) -> dict | None:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT j.*, COALESCE(l.size_bytes, 0) AS log_bytes, l.path AS log_path "
            "FROM jobs j LEFT JOIN job_logs l ON l.job_id = j.job_id "
            "WHERE j.job_id = ?",
            (job_id,),
        ).fetchone()
    return _row_to_job(row) if row else None


def mark_seen(job_id: str) -> bool:
    """Record that the user has opened this job.

    Purely a dashboard-side note: it never touches the cluster, and it only
    ever stops a finished run from announcing itself.
    """
    with db.transaction() as connection:
        cursor = connection.execute(
            "UPDATE jobs SET seen_at = ? WHERE job_id = ?", (int(time.time()), job_id)
        )
    return cursor.rowcount > 0


def status() -> dict:
    now = int(time.time())
    with db.connect() as connection:
        agent = connection.execute("SELECT * FROM agent_status WHERE id = 1").fetchone()
        counts = connection.execute(
            "SELECT state, COUNT(*) AS n FROM jobs "
            "WHERE is_agent = 0 GROUP BY state"
        ).fetchall()
        sres = connection.execute("SELECT * FROM sres_snapshot WHERE id = 1").fetchone()

    agent_dict = dict(agent) if agent else {}
    last_heartbeat = agent_dict.get("last_heartbeat")
    return {
        "server_time": now,
        "agent": {
            **agent_dict,
            "last_heartbeat": last_heartbeat,
            "seconds_since_heartbeat": (
                now - last_heartbeat if last_heartbeat else None
            ),
        },
        "state_counts": {row["state"]: row["n"] for row in counts},

        "sres": dict(sres) if sres else None,
        "retention_days": RETENTION_DAYS,
    }


def prune(retention_days: int = RETENTION_DAYS) -> int:
    """Drop jobs that ended past the retention horizon, and their logs."""
    cutoff = int(time.time()) - retention_days * 86400
    with db.transaction() as connection:
        rows = connection.execute(
            "SELECT job_id FROM jobs WHERE end_ts IS NOT NULL AND end_ts < ?",
            (cutoff,),
        ).fetchall()
        job_ids = [row["job_id"] for row in rows]
        if job_ids:
            marks = ", ".join("?" for _ in job_ids)
            connection.execute(f"DELETE FROM jobs WHERE job_id IN ({marks})", job_ids)
            connection.execute(f"DELETE FROM job_logs WHERE job_id IN ({marks})", job_ids)

    for job_id in job_ids:
        logstore.delete(job_id)
    return len(job_ids)
