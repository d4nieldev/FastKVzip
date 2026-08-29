"""Server tests: ingest offsets, window queries, hiding, retention."""

import base64
import gzip
import time

import pytest


def chunk(job_id: str, offset: int, text: str) -> dict:
    return {
        "job_id": job_id,
        "offset": offset,
        "encoding": "gzip+base64",
        "data": base64.b64encode(gzip.compress(text.encode())).decode(),
    }


def make_job(job_id="1001", **overrides) -> dict:
    job = {
        "job_id": job_id,
        "name": "graph-train",
        "user": "danieloh",
        "state": "RUNNING",
        "partition": "main",
        "time_limit_s": 3600,
        "elapsed_s": 600,
        "remaining_s": 3000,
        "submit_ts": 1_000_000,
        "start_ts": 1_000_100,
        "end_ts": None,
        "req_tres": "cpu=8,mem=60G,gres/gpu:rtx_pro_6000=1",
        "gres": "rtx_pro_6000:1",
    }
    job.update(overrides)
    return job


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_ingest_stores_jobs_and_logs(server):
    result = server.ingest.apply_payload(
        {
            "agent": {"job_id": "999", "host": "cs-cluster", "version": 1},
            "jobs": [make_job()],
            "logs": [chunk("1001", 0, "hello\n")],
        }
    )
    assert result["jobs_accepted"] == 1
    assert result["ack"] == {"1001": 6}
    assert result["reset"] == []

    job = server.queries.get_job("1001")
    assert job["state"] == "RUNNING"
    assert job["gres"] == "rtx_pro_6000:1"
    assert job["log_bytes"] == 6

    data, total = server.logstore.read_range("1001")
    assert data == b"hello\n" and total == 6


def test_sequential_chunks_append(server):
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "aaa")]})
    result = server.ingest.apply_payload({"jobs": [], "logs": [chunk("1001", 3, "bbb")]})
    assert result["ack"]["1001"] == 6
    assert server.logstore.read_range("1001")[0] == b"aaabbb"


def test_duplicate_chunk_is_idempotent(server):
    """A retried poll after a timed-out response must not double-write."""
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "aaa")]})
    result = server.ingest.apply_payload({"jobs": [], "logs": [chunk("1001", 0, "aaa")]})
    assert result["ack"]["1001"] == 3
    assert server.logstore.read_range("1001")[0] == b"aaa"


def test_partially_overlapping_chunk_is_trimmed(server):
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "aaa")]})
    server.ingest.apply_payload({"jobs": [], "logs": [chunk("1001", 1, "aabbb")]})
    assert server.logstore.read_range("1001")[0] == b"aaabbb"


def test_gap_triggers_reset_rather_than_corrupt_log(server):
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "aaa")]})
    result = server.ingest.apply_payload({"jobs": [], "logs": [chunk("1001", 500, "zzz")]})
    assert result["reset"] == ["1001"]
    assert result["ack"]["1001"] == 3
    assert server.logstore.read_range("1001")[0] == b"aaa"


def test_wiped_disk_asks_the_agent_to_reship_from_zero(server):
    """The ephemeral free-tier case: the store is empty but the agent is at 5000."""
    result = server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 5000, "x")]})
    assert result["reset"] == ["1001"]
    assert result["ack"]["1001"] == 0


def test_wiped_disk_is_detected_even_when_the_log_is_quiet(server):
    """A job producing no new output ships no chunks, so the declared offsets
    are the only thing that can reveal a lost log."""
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "hello\n")]})
    server.logstore.delete("1001")

    result = server.ingest.apply_payload({"jobs": [make_job()], "log_offsets": {"1001": 6}})
    assert result["reset"] == ["1001"]
    assert result["ack"]["1001"] == 0


def test_declared_offset_matching_the_store_asks_for_nothing(server):
    server.ingest.apply_payload({"jobs": [make_job()], "logs": [chunk("1001", 0, "hello\n")]})
    result = server.ingest.apply_payload({"jobs": [], "log_offsets": {"1001": 6}})
    assert result["reset"] == []
    assert result["ack"] == {}


def test_truncate_replaces_a_rewritten_log(server):
    """A requeued job overwrites its log with something shorter; without the
    truncate flag that would look like an already-stored duplicate."""
    server.ingest.apply_payload(
        {"jobs": [make_job()], "logs": [chunk("1001", 0, "first attempt, quite long\n")]}
    )
    result = server.ingest.apply_payload(
        {"jobs": [], "logs": [{**chunk("1001", 0, "retry\n"), "truncate": True}]}
    )
    assert result["ack"]["1001"] == 6
    assert server.logstore.read_range("1001")[0] == b"retry\n"


def test_traversal_job_ids_are_refused(server):
    result = server.ingest.apply_payload(
        {"jobs": [make_job("../../etc/passwd")], "logs": [chunk("../../etc/passwd", 0, "x")]}
    )
    assert result["jobs_accepted"] == 0
    assert result["ack"] == {}
    with pytest.raises(server.logstore.UnsafeJobId):
        server.logstore.log_path("../../etc/passwd")


def test_corrupt_chunk_triggers_reset(server):
    server.ingest.apply_payload({"jobs": [make_job()]})
    result = server.ingest.apply_payload(
        {"logs": [{"job_id": "1001", "offset": 0, "encoding": "gzip+base64", "data": "!!!"}]}
    )
    assert result["reset"] == ["1001"]


def test_reingest_updates_state_but_preserves_hidden(server):
    server.ingest.apply_payload({"jobs": [make_job(state="FAILED", end_ts=1_000_700)]})
    server.queries.set_hidden("1001", True)

    # The agent keeps reporting this job until sacct ages it out; the user's
    # decision to dismiss it must survive every one of those polls.
    server.ingest.apply_payload({"jobs": [make_job(state="FAILED", end_ts=1_000_700)]})
    assert server.queries.get_job("1001")["hidden"] is True


def test_first_seen_is_not_overwritten_on_reingest(server):
    server.ingest.apply_payload({"jobs": [make_job()]})
    first = server.queries.get_job("1001")["first_seen"]
    time.sleep(1.1)
    server.ingest.apply_payload({"jobs": [make_job()]})
    job = server.queries.get_job("1001")
    assert job["first_seen"] == first
    assert job["last_seen"] > first


# --------------------------------------------------------------------------- #
# Window query
# --------------------------------------------------------------------------- #


@pytest.fixture()
def windowed(server):
    now = int(time.time())
    hour = 3600
    server.ingest.apply_payload(
        {
            "jobs": [
                # Finished 5 hours ago.
                make_job("100", state="COMPLETED", submit_ts=now - 7 * hour,
                         start_ts=now - 6 * hour, end_ts=now - 5 * hour),
                # Finished 30 minutes ago.
                make_job("200", state="FAILED", submit_ts=now - 2 * hour,
                         start_ts=now - hour, end_ts=now - hour // 2),
                # Still running, started 4 hours ago.
                make_job("300", state="RUNNING", submit_ts=now - 5 * hour,
                         start_ts=now - 4 * hour, end_ts=None),
                # Queued, never started.
                make_job("400", state="PENDING", submit_ts=now - 10 * 60,
                         start_ts=None, end_ts=None),
            ]
        }
    )
    return server, now, hour


def test_window_selects_by_overlap_not_containment(windowed):
    server, now, hour = windowed
    ids = {job["job_id"] for job in server.queries.list_jobs(window_from=now - hour, window_to=now)}
    # 300 started before the window and is still going -- it was alive inside it.
    assert ids == {"200", "300", "400"}


def test_window_excludes_jobs_that_ended_before_it(windowed):
    server, now, hour = windowed
    ids = {job["job_id"] for job in server.queries.list_jobs(window_from=now - hour, window_to=now)}
    assert "100" not in ids


def test_wider_window_includes_older_jobs(windowed):
    server, now, hour = windowed
    ids = {job["job_id"] for job in server.queries.list_jobs(window_from=now - 24 * hour, window_to=now)}
    assert ids == {"100", "200", "300", "400"}


def test_window_excludes_jobs_that_started_after_it(windowed):
    server, now, hour = windowed
    ids = {job["job_id"] for job in server.queries.list_jobs(window_from=now - 8 * hour, window_to=now - 5 * hour)}
    # 300 was submitted before this window closes but did not start until after
    # it, so it was never running inside it.
    assert ids == {"100"}


def test_state_and_search_filters(windowed):
    server, now, hour = windowed
    failed = server.queries.list_jobs(states=["FAILED"])
    assert [job["job_id"] for job in failed] == ["200"]
    assert failed[0]["is_failure"] is True
    assert server.queries.list_jobs(search="nothing-matches") == []
    assert len(server.queries.list_jobs(search="graph-train")) == 4


def test_hidden_jobs_are_excluded_by_default(windowed):
    server, _, _ = windowed
    assert server.queries.set_hidden("200", True) is True
    assert "200" not in {job["job_id"] for job in server.queries.list_jobs()}
    assert "200" in {job["job_id"] for job in server.queries.list_jobs(include_hidden=True)}

    server.queries.set_hidden("200", False)
    assert "200" in {job["job_id"] for job in server.queries.list_jobs()}


def test_active_jobs_sort_ahead_of_finished_ones(windowed):
    server, _, _ = windowed
    ordered = [job["job_id"] for job in server.queries.list_jobs()]
    assert set(ordered[:2]) == {"300", "400"}


def test_set_hidden_on_unknown_job_reports_failure(server):
    assert server.queries.set_hidden("does-not-exist", True) is False


# --------------------------------------------------------------------------- #
# Status and retention
# --------------------------------------------------------------------------- #


def test_status_reports_heartbeat_and_counts(server):
    server.ingest.apply_payload(
        {
            "agent": {"job_id": "999", "host": "cs-cluster", "poll_interval": 30},
            "jobs": [make_job("1"), make_job("2", state="FAILED", end_ts=1_000_700)],
            "sres": "gpu01 rtx_pro_6000 free",
        }
    )
    status = server.queries.status()
    assert status["agent"]["job_id"] == "999"
    assert status["agent"]["seconds_since_heartbeat"] < 5
    assert status["state_counts"] == {"RUNNING": 1, "FAILED": 1}
    assert "rtx_pro_6000" in status["sres"]["body"]


def test_status_without_an_agent_reports_no_heartbeat(server):
    status = server.queries.status()
    assert status["agent"]["seconds_since_heartbeat"] is None


def test_prune_drops_old_jobs_and_their_logs(server):
    old = int(time.time()) - 40 * 86400
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("old", state="COMPLETED", end_ts=old),
                make_job("recent", state="COMPLETED", end_ts=int(time.time()) - 3600),
                make_job("running", state="RUNNING", end_ts=None),
            ],
            "logs": [chunk("old", 0, "stale"), chunk("recent", 0, "fresh")],
        }
    )
    assert server.queries.prune(retention_days=30) == 1

    remaining = {job["job_id"] for job in server.queries.list_jobs()}
    assert remaining == {"recent", "running"}
    assert server.logstore.current_size("old") == 0
    assert server.logstore.current_size("recent") == 5


def test_prune_never_removes_running_jobs(server):
    server.ingest.apply_payload({"jobs": [make_job("running", state="RUNNING", end_ts=None)]})
    assert server.queries.prune(retention_days=0) == 0
    assert server.queries.get_job("running") is not None


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(server, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    from app import main

    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client


def test_ingest_endpoint_requires_the_token(client):
    body = {"jobs": [make_job()]}
    assert client.post("/api/ingest", json=body).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"X-Agent-Token": "test-token"}).status_code == 200


def test_ingest_endpoint_accepts_gzipped_bodies(client):
    import json

    body = gzip.compress(json.dumps({"jobs": [make_job()], "logs": [chunk("1001", 0, "hi")]}).encode())
    response = client.post(
        "/api/ingest",
        content=body,
        headers={"X-Agent-Token": "test-token", "Content-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.json()["ack"] == {"1001": 2}


def test_read_endpoints_need_no_auth(client):
    client.post("/api/ingest", json={"jobs": [make_job()], "logs": [chunk("1001", 0, "line one\n")]},
                headers={"X-Agent-Token": "test-token"})

    assert client.get("/healthz").json()["ok"] is True
    assert len(client.get("/api/jobs").json()["jobs"]) == 1
    assert client.get("/api/jobs/1001").json()["state"] == "RUNNING"

    log = client.get("/api/jobs/1001/log").json()
    assert log["text"] == "line one\n"
    assert log["total_size"] == 9 and log["next_offset"] == 9

    assert client.get("/api/jobs/1001/log/download").text == "line one\n"


def test_log_endpoint_supports_tail_and_offset(client):
    client.post("/api/ingest", json={"jobs": [make_job()], "logs": [chunk("1001", 0, "abcdefghij")]},
                headers={"X-Agent-Token": "test-token"})

    tail = client.get("/api/jobs/1001/log?tail=4").json()
    assert tail["text"] == "ghij" and tail["offset"] == 6

    forward = client.get("/api/jobs/1001/log?offset=6").json()
    assert forward["text"] == "ghij"


def test_hide_endpoints_toggle_visibility(client):
    client.post("/api/ingest", json={"jobs": [make_job(state="FAILED", end_ts=1_000_700)]},
                headers={"X-Agent-Token": "test-token"})

    assert client.post("/api/jobs/1001/hide").json()["hidden"] is True
    assert client.get("/api/jobs").json()["jobs"] == []
    assert len(client.get("/api/jobs?include_hidden=true").json()["jobs"]) == 1

    assert client.delete("/api/jobs/1001/hide").json()["hidden"] is False
    assert len(client.get("/api/jobs").json()["jobs"]) == 1


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/log").status_code == 404
    assert client.post("/api/jobs/does-not-exist/hide").status_code == 404
