"""Regression tests for fields the UI reads straight off the list response."""

from test_server import chunk, make_job


def test_list_includes_log_path_for_the_detail_panel(server):
    server.ingest.apply_payload(
        {
            "jobs": [make_job()],
            "logs": [{**chunk("1001", 0, "x"), "path": "/home/danieloh/.slurm/logs/1001-run.log"}],
        }
    )
    listed = server.queries.list_jobs()[0]
    # The detail view renders from the list response, so a field only present
    # in get_job would silently show as unknown.
    assert listed["log_path"] == "/home/danieloh/.slurm/logs/1001-run.log"
    assert listed["log_bytes"] == 1
    assert server.queries.get_job("1001")["log_path"] == listed["log_path"]


def test_pending_job_reports_no_start_time(server):
    """The UI keys 'queued' off start_ts, so it must stay null while pending."""
    server.ingest.apply_payload(
        {"jobs": [make_job("1003", state="PENDING", start_ts=None, end_ts=None,
                           elapsed_s=0, remaining_s=None, reason="Dependency")]}
    )
    job = server.queries.get_job("1003")
    assert job["start_ts"] is None
    assert job["elapsed_s"] == 0
    assert job["is_terminal"] is False
    assert job["reason"] == "Dependency"
