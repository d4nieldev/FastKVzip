"""Read queries and retention for the dashboard API."""

from __future__ import annotations

import os
import re
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


# Eight hues in a fixed order, stepped for this dashboard's dark surface. Taken
# from the documented categorical palette rather than picked by eye, and checked
# against the panel background: every slot clears 3:1, and adjacent pairs clear
# the colour-blind separation gate.
#
# The order matters. Any two tags can appear side by side -- a project's next to
# a user's on one card -- and no eight-colour set survives that test; the first
# three do. Assigning in this order therefore keeps the common case, a handful
# of users, genuinely distinct, and every tag carries its name as text so colour
# is never what identifies it.
PALETTE = [
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
]

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def normalize_color(value) -> str | None:
    """A caller-supplied colour, or None if it is not a plain hex triplet."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value.lower() if _HEX.match(value) else None


def _next_color(taken: list[str]) -> str:
    """The first unused slot, else whichever is least spoken for."""
    for color in PALETTE:
        if color not in taken:
            return color
    counts = {color: taken.count(color) for color in PALETTE}
    return min(PALETTE, key=lambda color: counts[color])


def ensure_colors(connection) -> None:
    """Give a colour to anyone who has appeared without one.

    Assigned here rather than left to the browser so it is the same on a phone
    as on a laptop, and assigned in palette order rather than at random so the
    first few users are the ones the palette keeps furthest apart.
    """
    known = {row["user"] for row in connection.execute("SELECT DISTINCT user FROM agents")}
    known |= {
        row["user"]
        for row in connection.execute(
            f"SELECT DISTINCT user FROM jobs WHERE user IS NOT NULL AND {_not_agent()}"
        )
    }
    have = {row["user"]: row["color"] for row in connection.execute("SELECT * FROM user_colors")}
    taken = list(have.values())
    for user in sorted(known - set(have)):
        color = _next_color(taken)
        taken.append(color)
        connection.execute(
            "INSERT OR IGNORE INTO user_colors (user, color) VALUES (?, ?)", (user, color)
        )

    uncoloured = [
        row["id"]
        for row in connection.execute("SELECT id FROM projects WHERE color IS NULL ORDER BY created_at")
    ]
    if uncoloured:
        project_taken = [
            row["color"]
            for row in connection.execute("SELECT color FROM projects WHERE color IS NOT NULL")
        ]
        for project_id in uncoloured:
            color = _next_color(project_taken)
            project_taken.append(color)
            connection.execute("UPDATE projects SET color = ? WHERE id = ?", (color, project_id))


def set_user_color(user: str, color: str) -> bool:
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO user_colors (user, color) VALUES (?, ?) "
            "ON CONFLICT(user) DO UPDATE SET color = excluded.color",
            (user, color),
        )
    return True


def set_project_color(project_id: str, color: str) -> bool:
    with db.transaction() as connection:
        cursor = connection.execute(
            "UPDATE projects SET color = ? WHERE id = ?", (color, project_id)
        )
    return cursor.rowcount > 0


def _not_agent(alias: str = "") -> str:
    """SQL for "this is somebody's experiment, not a dashboard agent".

    Shared so the list, the state counts and the per-user summary cannot drift
    apart on what they consider a real job. Agents are matched by name as well
    as by the flag, because a retired one comes back from sacct unflagged.
    """
    prefix = f"{alias}." if alias else ""
    return (
        f"{prefix}is_agent = 0 AND ({prefix}name IS NULL OR {prefix}name NOT IN "
        "(SELECT name FROM jobs WHERE is_agent = 1 AND name IS NOT NULL))"
    )


# Job id descending stays the default -- ids grow over time, so it is "newest
# first" without depending on any timestamp being set. Every ordering ends with
# it, so equal keys never shuffle between polls.
_BY_ID = "CAST(j.job_id AS INTEGER) DESC, CAST(SUBSTR(j.job_id, INSTR(j.job_id, '_') + 1) AS INTEGER) DESC"

# Active work first, then what needs attention, then what is merely done. The
# alphabet would put CANCELLED above RUNNING, which is nobody's priority.
_STATE_RANK = (
    "CASE j.state "
    "WHEN 'RUNNING' THEN 0 WHEN 'COMPLETING' THEN 0 "
    "WHEN 'PENDING' THEN 1 WHEN 'CONFIGURING' THEN 1 WHEN 'SUSPENDED' THEN 1 "
    "WHEN 'FAILED' THEN 2 WHEN 'TIMEOUT' THEN 2 WHEN 'OUT_OF_MEMORY' THEN 2 "
    "WHEN 'NODE_FAIL' THEN 2 WHEN 'BOOT_FAIL' THEN 2 WHEN 'DEADLINE' THEN 2 "
    "WHEN 'PREEMPTED' THEN 3 WHEN 'CANCELLED' THEN 3 "
    "WHEN 'COMPLETED' THEN 4 ELSE 5 END"
)

# Each ordering is an expression plus the direction that is useful by default:
# newest first for a timestamp, active first for state, A to Z for a name. The
# caller may reverse any of them.
SORTS = {
    "id": ("CAST(j.job_id AS INTEGER)", "desc"),
    "state": (_STATE_RANK, "asc"),
    "submitted": ("j.submit_ts", "desc"),
    "started": ("j.start_ts", "desc"),
    "ended": ("j.end_ts", "desc"),
    "runtime": ("j.elapsed_s", "desc"),
    "name": ("j.name COLLATE NOCASE", "asc"),
}
DEFAULT_SORT = "id"


def order_by(sort: str | None, direction: str | None) -> str:
    """The ORDER BY for one of the offered sorts.

    Unknowns fall back rather than reaching SQL: both arrive from a query
    string, and a stale bookmark should not be able to break the page or say
    anything the database will run.

    Whatever is unknown sorts last in either direction -- a job that never
    started belongs at the bottom of a start-time list, not at the top of the
    ascending one. Every ordering ends on the job id so equal keys cannot
    shuffle between polls.
    """
    expression, default = SORTS.get(sort or "", SORTS[DEFAULT_SORT])
    chosen = (direction or default).lower()
    keyword = "ASC" if chosen == "asc" else "DESC"
    return f"({expression}) IS NULL, {expression} {keyword}, {_BY_ID}"


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
    users: list[str] | None = None,
    project: str | None = None,
    sort: str | None = None,
    direction: str | None = None,
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
    if users:
        clauses.append(f"j.user IN ({', '.join('?' for _ in users)})")
        params.extend(users)
    if project is not None:
        clauses.append("j.project_id = ?")
        params.append(project)
    if not include_agent:
        # An agent job is infrastructure, not an experiment. Its health is the
        # banner at the top of the page, which is where it can still be opened
        # from; in the list it is one more row to read past.
        clauses.append(_not_agent("j"))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        # log_path is included here as well as in get_job: the detail panel
        # renders straight from the list response rather than refetching.
        "SELECT j.*, COALESCE(l.size_bytes, 0) AS log_bytes, l.path AS log_path, "
        "uc.color AS user_color, p.name AS project_name, p.color AS project_color "
        "FROM jobs j LEFT JOIN job_logs l ON l.job_id = j.job_id "
        "LEFT JOIN user_colors uc ON uc.user = j.user "
        "LEFT JOIN projects p ON p.id = j.project_id "
        f"{where} "
        f"ORDER BY {order_by(sort, direction)}"
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


def mark_seen_many(job_ids: list[str]) -> int:
    """Mark a batch of jobs read in one write.

    Takes the ids rather than clearing everything unseen: the button clears
    what is on screen, and a run outside the current window or filter should
    not be quietly marked read on the strength of a click that never showed it.
    """
    if not job_ids:
        return 0
    now = int(time.time())
    placeholders = ", ".join("?" for _ in job_ids)
    with db.transaction() as connection:
        cursor = connection.execute(
            f"UPDATE jobs SET seen_at = ? WHERE job_id IN ({placeholders})",
            [now, *job_ids],
        )
    return cursor.rowcount


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


def user_summaries(connection, now: int) -> list[dict]:
    """One entry per user the dashboard knows about.

    The union of who is reporting and whose jobs are on file, so a user whose
    agent has stopped still appears while their history is retained, and one
    whose agent has only just started appears before their first job lands.
    """
    agents = connection.execute("SELECT * FROM agents").fetchall()
    colors = {row["user"]: row["color"] for row in connection.execute("SELECT * FROM user_colors")}
    counts = connection.execute(
        f"SELECT user, state, COUNT(*) AS n FROM jobs WHERE {_not_agent()} "
        "GROUP BY user, state"
    ).fetchall()
    terminal = ", ".join("?" for _ in TERMINAL_STATES)
    unseen = connection.execute(
        f"SELECT user, COUNT(*) AS n FROM jobs WHERE {_not_agent()} "
        f"AND state IN ({terminal}) "
        "AND (seen_at IS NULL OR (end_ts IS NOT NULL AND seen_at < end_ts)) "
        "GROUP BY user",
        sorted(TERMINAL_STATES),
    ).fetchall()

    by_user: dict[str, dict] = {}

    def entry_for(user: str) -> dict:
        return by_user.setdefault(
            user,
            {
                "user": user,
                "last_heartbeat": None,
                "seconds_since_heartbeat": None,
                "job_id": None,
                "host": None,
                "version": None,
                "poll_interval": None,
                "cluster_time": None,
                "state_counts": {},
                "unseen_count": 0,
                "color": None,
            },
        )

    for row in agents:
        if not row["user"]:
            continue
        entry = entry_for(row["user"])
        entry.update({key: row[key] for key in row.keys()})
        last = entry.get("last_heartbeat")
        entry["seconds_since_heartbeat"] = now - last if last else None

    for row in counts:
        if row["user"]:
            entry_for(row["user"])["state_counts"][row["state"]] = row["n"]
    for row in unseen:
        if row["user"]:
            entry_for(row["user"])["unseen_count"] = row["n"]

    for user, entry in by_user.items():
        entry["color"] = colors.get(user)

    # Reporting users first, then by name, so a live agent never sorts below a
    # user whose history merely lingers.
    return sorted(
        by_user.values(),
        key=lambda entry: (entry["last_heartbeat"] is None, entry["user"]),
    )


def status() -> dict:
    now = int(time.time())
    with db.transaction() as connection:
        ensure_colors(connection)

    with db.connect() as connection:
        users = user_summaries(connection, now)
        projects = project_summaries(connection, now)
        counts = connection.execute(
            f"SELECT state, COUNT(*) AS n FROM jobs WHERE {_not_agent()} GROUP BY state"
        ).fetchall()
        sres = connection.execute("SELECT * FROM sres_snapshot WHERE id = 1").fetchone()

    return {
        "server_time": now,
        "users": users,
        "projects": projects,
        "state_counts": {row["state"]: row["n"] for row in counts},
        "sres": dict(sres) if sres else None,
        "retention_days": RETENTION_DAYS,
    }


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #

_SLUG_TRIM = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """A short, url-safe id from a project name.

    Ids appear in the address bar and in whatever an automated submitter writes
    into its own logs, so they are readable rather than random.
    """
    slug = _SLUG_TRIM.sub("-", name.strip().lower()).strip("-")[:64]
    return slug or "project"


def create_project(name: str, project_id: str | None = None) -> dict:
    """Create a project, or return the existing one with that id.

    Idempotent by id on purpose: an automated submitter can call this at the
    start of every run without having to remember whether it already has.
    """
    name = name.strip()[:200] or "Untitled"
    wanted = slugify(project_id or name)
    now = int(time.time())

    with db.transaction() as connection:
        existing = connection.execute(
            "SELECT * FROM projects WHERE id = ?", (wanted,)
        ).fetchone()
        if existing:
            return dict(existing)
        connection.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (wanted, name, now),
        )
    return {"id": wanted, "name": name, "created_at": now}


def assign_jobs(project_id: str | None, job_ids: list[str]) -> int:
    """Put jobs in a project, or take them out of one when project_id is None.

    Unknown ids are ignored rather than refused: a submitter that hands over a
    whole grid should not have the call fail because one id has since been
    pruned.
    """
    if not job_ids:
        return 0
    placeholders = ", ".join("?" for _ in job_ids)
    with db.transaction() as connection:
        if project_id is not None and not connection.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone():
            raise KeyError(project_id)
        cursor = connection.execute(
            f"UPDATE jobs SET project_id = ? WHERE job_id IN ({placeholders})",
            [project_id, *job_ids],
        )
    return cursor.rowcount


def delete_project(project_id: str) -> bool:
    """Remove a project. Its jobs stay; they simply belong to nothing again."""
    with db.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET project_id = NULL WHERE project_id = ?", (project_id,)
        )
        cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cursor.rowcount > 0


def project_summaries(connection, now: int) -> list[dict]:
    """Every project, with enough to choose between them without opening one."""
    projects = connection.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    if not projects:
        return []

    counts = connection.execute(
        f"SELECT project_id, state, COUNT(*) AS n FROM jobs "
        f"WHERE {_not_agent()} AND project_id IS NOT NULL GROUP BY project_id, state"
    ).fetchall()
    owners = connection.execute(
        f"SELECT DISTINCT project_id, user FROM jobs "
        f"WHERE {_not_agent()} AND project_id IS NOT NULL AND user IS NOT NULL"
    ).fetchall()
    terminal = ", ".join("?" for _ in TERMINAL_STATES)
    unseen = connection.execute(
        f"SELECT project_id, COUNT(*) AS n FROM jobs WHERE {_not_agent()} "
        f"AND project_id IS NOT NULL AND state IN ({terminal}) "
        "AND (seen_at IS NULL OR (end_ts IS NOT NULL AND seen_at < end_ts)) "
        "GROUP BY project_id",
        sorted(TERMINAL_STATES),
    ).fetchall()
    activity = connection.execute(
        f"SELECT project_id, MAX(COALESCE(end_ts, start_ts, submit_ts, first_seen)) AS t "
        f"FROM jobs WHERE {_not_agent()} AND project_id IS NOT NULL GROUP BY project_id"
    ).fetchall()

    by_id = {}
    for row in projects:
        by_id[row["id"]] = {
            **dict(row),
            "state_counts": {},
            "users": [],
            "unseen_count": 0,
            "last_activity": None,
            "job_count": 0,
        }
    for row in counts:
        entry = by_id.get(row["project_id"])
        if entry:
            entry["state_counts"][row["state"]] = row["n"]
            entry["job_count"] += row["n"]
    for row in owners:
        entry = by_id.get(row["project_id"])
        if entry:
            entry["users"].append(row["user"])
    for row in unseen:
        entry = by_id.get(row["project_id"])
        if entry:
            entry["unseen_count"] = row["n"]
    for row in activity:
        entry = by_id.get(row["project_id"])
        if entry:
            entry["last_activity"] = row["t"]

    for entry in by_id.values():
        entry["users"].sort()

    # Most recently active first; a project nobody has run yet sorts by
    # creation, which is all it has.
    return sorted(
        by_id.values(),
        key=lambda entry: (entry["last_activity"] or entry["created_at"]),
        reverse=True,
    )


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
