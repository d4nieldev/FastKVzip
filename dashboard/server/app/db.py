"""SQLite storage for job records, agent heartbeat and the sres snapshot.

Log bodies live on disk (see logstore), not in the database -- byte-range reads
and whole-file downloads then cost nothing.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "dashboard.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    name          TEXT,
    user          TEXT,
    state         TEXT NOT NULL,
    partition     TEXT,
    reason        TEXT,
    dependency    TEXT,
    exit_code     TEXT,
    submit_ts     INTEGER,
    start_ts      INTEGER,
    est_start_ts  INTEGER,
    end_ts        INTEGER,
    elapsed_s     INTEGER,
    time_limit_s  INTEGER,
    remaining_s   INTEGER,
    cpus          TEXT,
    nodes         TEXT,
    node_list     TEXT,
    req_tres      TEXT,
    alloc_tres    TEXT,
    gres          TEXT,
    mem_req       TEXT,
    max_rss       TEXT,
    work_dir      TEXT,
    is_agent      INTEGER NOT NULL DEFAULT 0,
    first_seen    INTEGER NOT NULL,
    last_seen     INTEGER NOT NULL,
    -- When the user last opened this job. Compared against end_ts, not merely
    -- checked for null: opening a job while it ran says nothing about having
    -- seen how it ended.
    seen_at       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state);
CREATE INDEX IF NOT EXISTS idx_jobs_end ON jobs (end_ts);
CREATE INDEX IF NOT EXISTS idx_jobs_start ON jobs (start_ts);

-- The script a job was submitted with, and the environment it was submitted
-- from. Kept apart from `jobs` because they are large, immutable, and read
-- only when someone opens the panel -- never on the list query.
CREATE TABLE IF NOT EXISTS job_scripts (
    job_id        TEXT PRIMARY KEY,
    batch_script  TEXT,
    job_env       TEXT,
    -- Which SLURM source answered, so the panel can say whether it is showing
    -- the submitted script or the file as it stands on disk now.
    script_source TEXT,
    env_source    TEXT,
    note          TEXT,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS job_logs (
    job_id      TEXT PRIMARY KEY,
    path        TEXT,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_status (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    last_heartbeat INTEGER,
    job_id         TEXT,
    host           TEXT,
    user           TEXT,
    version        INTEGER,
    poll_interval  INTEGER,
    cluster_time   INTEGER
);

CREATE TABLE IF NOT EXISTS sres_snapshot (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    body       TEXT,
    updated_at INTEGER
);
"""

# Every column the agent may supply. Kept explicit so an unexpected key in a
# payload can never reach the SQL builder.
JOB_COLUMNS = (
    "name",
    "user",
    "state",
    "partition",
    "reason",
    "dependency",
    "exit_code",
    "submit_ts",
    "start_ts",
    "est_start_ts",
    "end_ts",
    "elapsed_s",
    "time_limit_s",
    "remaining_s",
    "cpus",
    "nodes",
    "node_list",
    "req_tres",
    "alloc_tres",
    "gres",
    "mem_req",
    "max_rss",
    "work_dir",
    "is_agent",
)


def connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        _migrate(connection)


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so a volume
    that survives a redeploy would otherwise keep the old shape and every
    insert naming a new column would fail.
    """
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
    for column, ddl in (("est_start_ts", "INTEGER"), ("seen_at", "INTEGER")):
        if column not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

    # Dismissal is gone: hiding a run made it unreachable, which is the
    # opposite of what a dashboard is for. Dropping the columns is tidiness,
    # not a requirement -- DROP COLUMN needs SQLite 3.35, so an older library
    # just leaves them sitting there unread.
    for column in ("hidden", "hidden_at"):
        if column in existing:
            try:
                connection.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
            except sqlite3.OperationalError:
                pass


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    connection = connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
