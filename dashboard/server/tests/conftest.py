import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
for path in (ROOT, os.path.join(REPO_ROOT, "dashboard", "agent")):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """A fresh server package bound to an isolated DATA_DIR.

    The storage paths are read at import time, so the modules are reloaded per
    test rather than shared -- otherwise every test would hit the same file.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-token")

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
    return bundle
