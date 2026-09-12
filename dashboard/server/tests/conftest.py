import importlib
import os
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
for path in (ROOT, os.path.join(REPO_ROOT, "dashboard", "agent")):
    if path not in sys.path:
        sys.path.insert(0, path)

# The suite runs against a real Postgres, because that is what the dashboard
# runs against. A compatibility layer would have let the SQL that ships go
# untested -- which is the whole reason this storage exists.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:dev@127.0.0.1:55432/dashboard"
)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """A fresh server package bound to a schema of its own.

    A schema rather than a database: creating one is cheap enough to do per
    test, and dropping it cascades, so no test can see another's rows.
    """
    schema = f"t{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-token")
    monkeypatch.setenv(
        "DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema}"
    )

    import psycopg

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as setup:
        setup.execute(f"CREATE SCHEMA {schema}")

    from app import db, ingest, logstore, queries

    for module in (db, logstore, ingest, queries):
        importlib.reload(module)
    db.init_db()

    class Server:
        pass

    bundle = Server()
    bundle.db = db
    bundle.logstore = logstore
    bundle.ingest = ingest
    bundle.queries = queries
    try:
        yield bundle
    finally:
        if db._POOL is not None:
            db._POOL.close()
            db._POOL = None
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as teardown:
            teardown.execute(f"DROP SCHEMA {schema} CASCADE")
