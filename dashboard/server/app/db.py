"""Postgres storage for job records, agent heartbeat and the sres snapshot.

Log bodies live on disk (see logstore), not in the database -- byte-range reads
and whole-file downloads then cost nothing, and a lost disk costs nothing
either because the agent re-ships what the server no longer holds. That is what
keeps this database small: everything in it is either tiny or irreplaceable.

Irreplaceable is the point. Jobs, logs and the sres snapshot all come back on
their own after a wipe, but projects, the jobs filed into them, chosen colours
and read marks exist nowhere else -- so they need storage that outlives the
container, which an ephemeral filesystem is not.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Where log files go. Still a filesystem, still allowed to be ephemeral.
DATA_DIR = os.environ.get("DATA_DIR", "/data")

DATABASE_URL = os.environ.get("DATABASE_URL", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    name          TEXT,
    "user"          TEXT,
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
    seen_at       INTEGER,
    -- The experiment this job belongs to, if anyone has said. Set by hand or by
    -- whatever submitted the job; never by the agent, which only reports what
    -- SLURM knows and would otherwise wipe it on every poll.
    project_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state);
CREATE INDEX IF NOT EXISTS idx_jobs_end ON jobs (end_ts);
CREATE INDEX IF NOT EXISTS idx_jobs_start ON jobs (start_ts);

CREATE TABLE IF NOT EXISTS job_logs (
    job_id      TEXT PRIMARY KEY,
    path        TEXT,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    updated_at  INTEGER NOT NULL
);

-- One row per agent, keyed by the user it reports for. Was a single row: with
-- several people running an agent, whichever polled last overwrote the rest.
CREATE TABLE IF NOT EXISTS agents (
    "user"           TEXT PRIMARY KEY,
    last_heartbeat INTEGER,
    job_id         TEXT,
    host           TEXT,
    version        INTEGER,
    poll_interval  INTEGER,
    cluster_time   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs ("user");
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs (project_id);

-- A named collection of jobs, cutting across users: one experiment is often
-- run by more than one person, and the submission batches that make it up
-- arrive hours apart.
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    color       TEXT
);

-- A colour per user, so a tag is recognisable before it is read. Its own table
-- rather than a column on `agents`: a user with jobs but no running agent has
-- no row there, and would lose their colour the moment their agent stopped.
CREATE TABLE IF NOT EXISTS user_colors (
    "user"   TEXT PRIMARY KEY,
    color  TEXT NOT NULL
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


_POOL: ConnectionPool | None = None


def pool() -> ConnectionPool:
    """The shared connection pool, opened on first use.

    Small on purpose: one agent posting every thirty seconds and a handful of
    browsers polling is not a load, and a free-tier Postgres counts connections
    far more jealously than queries.
    """
    global _POOL
    if _POOL is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is unset. The dashboard stores projects, colours "
                "and read marks, none of which anything else can replace, so it "
                "needs a database that outlives the container."
            )
        _POOL = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=int(os.environ.get("DB_POOL_SIZE", "5")),
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=True,
        )
    return _POOL


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """A pooled connection. Committed on a clean exit, rolled back otherwise."""
    with pool().connection() as connection:
        yield connection


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with connect() as connection:
        # psycopg runs several statements in one execute() as long as none of
        # them carries a parameter, which the schema does not.
        connection.execute(SCHEMA)
        _migrate(connection)


def _columns(connection, table: str) -> set[str]:
    return {
        row["column_name"]
        for row in connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
    }


def _migrate(connection: psycopg.Connection) -> None:
    """Add columns introduced after a database was first created.

    CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so a volume
    that survives a redeploy would otherwise keep the old shape and every
    insert naming a new column would fail.
    """
    existing = _columns(connection, "jobs")
    for column, ddl in (
        ("est_start_ts", "INTEGER"),
        ("seen_at", "INTEGER"),
        ("project_id", "TEXT"),
    ):
        if column not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

    # agent_status held one row for one agent. Nothing in it is worth keeping --
    # a heartbeat is replaced within a poll of the agent starting again.
    connection.execute("DROP TABLE IF EXISTS agent_status")

    if "color" not in _columns(connection, "projects"):
        connection.execute("ALTER TABLE projects ADD COLUMN color TEXT")

    # The submitted-script panel is gone: on a cluster that stores no scripts
    # in accounting it could only ever have covered jobs still running, which
    # is not the question anyone opens a finished job to ask.
    connection.execute("DROP TABLE IF EXISTS job_scripts")

    # Dismissal is gone: hiding a run made it unreachable, which is the
    # opposite of what a dashboard is for.
    for column in ("hidden", "hidden_at"):
        if column in existing:
            connection.execute(f'ALTER TABLE jobs DROP COLUMN "{column}"')


@contextmanager
def transaction() -> Iterator[psycopg.Connection]:
    """A write. Same thing as connect(); named apart so call sites read clearly."""
    with pool().connection() as connection:
        yield connection
