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



def test_jobs_are_ordered_by_job_id_descending(windowed):
    server, _, _ = windowed
    ordered = [job["job_id"] for job in server.queries.list_jobs()]
    assert ordered == ["400", "300", "200", "100"]


def test_job_id_order_is_numeric_not_lexicographic(server):
    server.ingest.apply_payload(
        {"jobs": [make_job("9"), make_job("1000"), make_job("90")]}
    )
    assert [job["job_id"] for job in server.queries.list_jobs()] == ["1000", "90", "9"]


def test_array_tasks_sort_by_task_index_under_their_base_id(server):
    server.ingest.apply_payload(
        {"jobs": [make_job("500_2"), make_job("500_10"), make_job("499")]}
    )
    ordered = [job["job_id"] for job in server.queries.list_jobs()]
    assert ordered == ["500_10", "500_2", "499"]



# --------------------------------------------------------------------------- #
# Status and retention
# --------------------------------------------------------------------------- #


def test_status_reports_each_agent_under_its_own_user(server):
    server.ingest.apply_payload(
        {
            "agent": {"job_id": "999", "host": "cs-cluster", "poll_interval": 30,
                      "user": "danieloh"},
            "jobs": [make_job("1"), make_job("2", state="FAILED", end_ts=1_000_700)],
            "sres": "gpu01 rtx_pro_6000 free",
        }
    )
    status = server.queries.status()
    (entry,) = status["users"]
    assert entry["user"] == "danieloh"
    assert entry["job_id"] == "999"
    assert entry["seconds_since_heartbeat"] is not None
    assert status["state_counts"] == {"RUNNING": 1, "FAILED": 1}
    assert "rtx_pro_6000" in status["sres"]["body"]


def test_two_agents_do_not_overwrite_each_other(server):
    # The whole point: agent_status held one row, so whichever agent polled
    # last became "the" agent and the other vanished.
    server.ingest.apply_payload(
        {
            "agent": {"job_id": "900", "host": "cpu-01", "user": "danieloh"},
            "jobs": [make_job("1", user="danieloh")],
        }
    )
    server.ingest.apply_payload(
        {
            "agent": {"job_id": "901", "host": "cpu-02", "user": "someone"},
            "jobs": [make_job("2", user="someone", state="FAILED", end_ts=1_000_700)],
        }
    )
    users = {entry["user"]: entry for entry in server.queries.status()["users"]}
    assert set(users) == {"danieloh", "someone"}
    assert users["danieloh"]["job_id"] == "900"
    assert users["someone"]["job_id"] == "901"
    assert users["danieloh"]["state_counts"] == {"RUNNING": 1}
    assert users["someone"]["state_counts"] == {"FAILED": 1}
    assert users["someone"]["unseen_count"] == 1


def test_jobs_can_be_narrowed_to_chosen_users(server):
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("1", user="danieloh"),
                make_job("2", user="someone"),
                make_job("3", user="third"),
            ]
        }
    )
    listed = lambda users: {j["job_id"] for j in server.queries.list_jobs(users=users)}
    assert listed(["danieloh"]) == {"1"}
    assert listed(["danieloh", "third"]) == {"1", "3"}
    assert listed(None) == {"1", "2", "3"}


def test_a_user_appears_before_their_first_job_lands(server):
    # An agent that has just started has reported a heartbeat and nothing else;
    # the picker should still offer it.
    server.ingest.apply_payload({"agent": {"job_id": "900", "user": "newcomer"}, "jobs": []})
    (entry,) = server.queries.status()["users"]
    assert entry["user"] == "newcomer"
    assert entry["state_counts"] == {}


def test_a_user_survives_their_agent_stopping(server):
    # Jobs are retained for 30 days; a user whose agent died still has history
    # worth reading, so they stay in the list with no heartbeat.
    server.ingest.apply_payload({"jobs": [make_job("1", user="departed")]})
    (entry,) = server.queries.status()["users"]
    assert entry["user"] == "departed"
    assert entry["seconds_since_heartbeat"] is None
    assert entry["state_counts"] == {"RUNNING": 1}


def test_an_agent_naming_no_user_does_not_take_over_another_row(server):
    server.ingest.apply_payload({"agent": {"job_id": "900", "user": "danieloh"}, "jobs": []})
    server.ingest.apply_payload({"agent": {"job_id": "901"}, "jobs": []})
    users = {entry["user"] for entry in server.queries.status()["users"]}
    assert users == {"danieloh", "unknown"}


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



def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/log").status_code == 404
    assert client.post("/api/jobs/does-not-exist/hide").status_code == 404


# --------------------------------------------------------------------------- #
# The dashboard's own job
# --------------------------------------------------------------------------- #


def test_the_agent_job_is_kept_out_of_the_list_and_the_counts(server):
    server.ingest.apply_payload(
        {"jobs": [make_job("500"), make_job("501", name="dashboard-agent", is_agent=True)]}
    )
    assert [job["job_id"] for job in server.queries.list_jobs()] == ["500"]
    assert server.queries.status()["state_counts"] == {"RUNNING": 1}


def test_the_agent_job_is_still_reachable_on_its_own(server):
    # Excluded from the list, not from the database: the banner links to it,
    # and its log is the first thing to read when the agent misbehaves.
    server.ingest.apply_payload({"jobs": [make_job("501", is_agent=True)]})
    assert server.queries.get_job("501")["is_agent"] is True
    assert [job["job_id"] for job in server.queries.list_jobs(include_agent=True)] == ["501"]


def test_a_retired_agent_job_does_not_rejoin_the_list_after_a_handover(server):
    server.ingest.apply_payload({"jobs": [make_job("501", is_agent=True)]})
    # The successor reports its predecessor as an ordinary job, because
    # is_agent means "this job is me" to whichever agent is running.
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("501", state="COMPLETED", end_ts=1_000_700, is_agent=False),
                make_job("502", is_agent=True),
            ]
        }
    )
    assert server.queries.list_jobs() == []
    assert server.queries.get_job("501")["is_agent"] is True


# --------------------------------------------------------------------------- #
# Unseen finished runs
# --------------------------------------------------------------------------- #


def test_a_finished_run_is_unseen_until_it_is_opened(server):
    server.ingest.apply_payload({"jobs": [make_job(state="FAILED", end_ts=1_000_700)]})
    assert server.queries.get_job("1001")["unseen"] is True
    assert server.queries.mark_seen("1001") is True
    assert server.queries.get_job("1001")["unseen"] is False


def test_a_running_job_is_never_unseen(server):
    server.ingest.apply_payload({"jobs": [make_job(state="RUNNING")]})
    assert server.queries.get_job("1001")["unseen"] is False


def test_opening_a_job_while_it_ran_does_not_cover_how_it_ended(server):
    server.ingest.apply_payload({"jobs": [make_job(state="RUNNING")]})
    server.queries.mark_seen("1001")
    # seen_at is now; the job then fails at a later timestamp, which is news.
    server.ingest.apply_payload(
        {"jobs": [make_job(state="FAILED", end_ts=int(time.time()) + 60)]}
    )
    assert server.queries.get_job("1001")["unseen"] is True


def test_seen_survives_the_agent_re_reporting_the_job(server):
    server.ingest.apply_payload({"jobs": [make_job(state="FAILED", end_ts=1_000_700)]})
    server.queries.mark_seen("1001")
    server.ingest.apply_payload({"jobs": [make_job(state="FAILED", end_ts=1_000_700)]})
    assert server.queries.get_job("1001")["unseen"] is False


def test_marking_an_unknown_job_seen_reports_failure(server):
    assert server.queries.mark_seen("does-not-exist") is False



# --------------------------------------------------------------------------- #
# Marking a batch read
# --------------------------------------------------------------------------- #


def test_many_jobs_are_marked_read_in_one_call(server):
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("100", state="FAILED", end_ts=1_000_700),
                make_job("200", state="COMPLETED", end_ts=1_000_700),
                make_job("300", state="FAILED", end_ts=1_000_700),
            ]
        }
    )
    assert server.queries.mark_seen_many(["100", "200"]) == 2
    unseen = {job["job_id"] for job in server.queries.list_jobs() if job["unseen"]}
    # 300 was not passed, so it keeps glowing: the button clears what was on
    # screen, not everything the database happens to hold.
    assert unseen == {"300"}


def test_marking_an_empty_batch_read_touches_nothing(server):
    server.ingest.apply_payload({"jobs": [make_job("100", state="FAILED", end_ts=1_000_700)]})
    assert server.queries.mark_seen_many([]) == 0
    assert server.queries.get_job("100")["unseen"] is True


def test_unknown_ids_in_a_batch_are_ignored(server):
    server.ingest.apply_payload({"jobs": [make_job("100", state="FAILED", end_ts=1_000_700)]})
    assert server.queries.mark_seen_many(["100", "does-not-exist"]) == 1


def test_a_retired_agent_is_recognised_by_name_alone(server):
    # What an old agent reports: itself flagged, its predecessor not, because
    # is_agent only ever meant "this job is me".
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("300", name="dashboard-agent", state="RUNNING", is_agent=True),
                make_job("200", name="dashboard-agent", state="COMPLETED", end_ts=1_000_700),
                make_job("100", name="gd32-seed0", state="COMPLETED", end_ts=1_000_700),
            ]
        }
    )
    # The server can still tell, without waiting for a newer agent to be
    # deployed on the cluster.
    assert [job["job_id"] for job in server.queries.list_jobs()] == ["100"]
    assert server.queries.get_job("200") is not None


def test_a_job_merely_sharing_no_name_with_the_agent_is_unaffected(server):
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("300", name="dashboard-agent", state="RUNNING", is_agent=True),
                make_job("100", name=None, state="COMPLETED", end_ts=1_000_700),
            ]
        }
    )
    assert [job["job_id"] for job in server.queries.list_jobs()] == ["100"]


# --------------------------------------------------------------------------- #
# Recovering after the server's disk is wiped
# --------------------------------------------------------------------------- #


def test_a_server_holding_no_jobs_asks_for_history(server):
    # A redeploy on ephemeral storage leaves the agent pushing only its own
    # job every 30s, while its history sweep is up to five minutes away.
    response = server.ingest.apply_payload(
        {"jobs": [make_job("500", name="dashboard-agent", is_agent=True)]}
    )
    assert response["want_history"] is True


def test_a_server_with_jobs_does_not_keep_asking(server):
    server.ingest.apply_payload({"jobs": [make_job("100")]})
    response = server.ingest.apply_payload({"jobs": [make_job("100")]})
    assert response["want_history"] is False


def test_the_history_sweep_itself_is_not_asked_to_repeat(server):
    # Answering "still nothing" to a full refresh would loop it every poll.
    response = server.ingest.apply_payload({"jobs": [], "full_refresh": True})
    assert response["want_history"] is False


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #


def test_a_project_gathers_jobs_from_more_than_one_user(server):
    # The point of a project: one experiment run by two people, whose jobs the
    # user cut necessarily keeps apart.
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("1", user="danieloh"),
                make_job("2", user="guyzagor"),
                make_job("3", user="danieloh"),
            ]
        }
    )
    project = server.queries.create_project("Gate ablation")
    assert project["id"] == "gate-ablation"
    assert server.queries.assign_jobs("gate-ablation", ["1", "2"]) == 2

    listed = server.queries.list_jobs(project="gate-ablation")
    assert {job["job_id"] for job in listed} == {"1", "2"}
    assert {job["user"] for job in listed} == {"danieloh", "guyzagor"}


def test_creating_a_project_twice_returns_the_same_one(server):
    # An automated submitter calls this at the start of every run without
    # having to remember whether it already has.
    first = server.queries.create_project("Gate ablation")
    second = server.queries.create_project("Gate ablation")
    assert first["id"] == second["id"] == "gate-ablation"
    assert first["created_at"] == second["created_at"]


def test_assigning_to_an_unknown_project_is_refused(server):
    server.ingest.apply_payload({"jobs": [make_job("1")]})
    with pytest.raises(KeyError):
        server.queries.assign_jobs("no-such-project", ["1"])


def test_unknown_job_ids_are_skipped_rather_than_failing_the_batch(server):
    # A grid handed over whole should not fail because one id has been pruned.
    server.ingest.apply_payload({"jobs": [make_job("1")]})
    server.queries.create_project("Sweep")
    assert server.queries.assign_jobs("sweep", ["1", "gone"]) == 1


def test_a_job_can_be_taken_back_out_of_every_project(server):
    server.ingest.apply_payload({"jobs": [make_job("1")]})
    server.queries.create_project("Sweep")
    server.queries.assign_jobs("sweep", ["1"])
    assert server.queries.assign_jobs(None, ["1"]) == 1
    assert server.queries.list_jobs(project="sweep") == []


def test_deleting_a_project_keeps_its_jobs(server):
    server.ingest.apply_payload({"jobs": [make_job("1")]})
    server.queries.create_project("Sweep")
    server.queries.assign_jobs("sweep", ["1"])
    assert server.queries.delete_project("sweep") is True
    assert server.queries.get_job("1") is not None
    assert server.queries.get_job("1")["project_id"] is None
    assert server.queries.delete_project("sweep") is False


def test_the_agent_never_clears_a_project_by_re_reporting(server):
    # project_id is not the agent's to know; it must survive every poll the way
    # first_seen and seen_at do.
    server.ingest.apply_payload({"jobs": [make_job("1")]})
    server.queries.create_project("Sweep")
    server.queries.assign_jobs("sweep", ["1"])
    server.ingest.apply_payload({"jobs": [make_job("1", state="COMPLETED", end_ts=1_000_700)]})
    assert server.queries.get_job("1")["project_id"] == "sweep"


def test_project_summaries_describe_a_project_without_opening_it(server):
    server.ingest.apply_payload(
        {
            "jobs": [
                make_job("1", user="danieloh", state="FAILED", end_ts=1_000_700),
                make_job("2", user="guyzagor", state="RUNNING"),
            ]
        }
    )
    server.queries.create_project("Gate ablation")
    server.queries.assign_jobs("gate-ablation", ["1", "2"])
    (entry,) = server.queries.status()["projects"]
    assert entry["name"] == "Gate ablation"
    assert entry["job_count"] == 2
    assert entry["state_counts"] == {"FAILED": 1, "RUNNING": 1}
    assert entry["users"] == ["danieloh", "guyzagor"]
    assert entry["unseen_count"] == 1


def test_a_name_becomes_a_readable_id(server):
    assert server.queries.slugify("Qwen3 8B / gate ablation!") == "qwen3-8b-gate-ablation"
    assert server.queries.slugify("   ") == "project"
