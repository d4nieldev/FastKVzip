"""Tests for the agent's self-resubmission, which keeps the chain alive."""

import os

import probe_agent as agent


def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(agent, "RESUBMIT_LOCK_PATH", str(tmp_path / "resubmit.lock"))


def test_successor_is_chained_after_this_job(tmp_path, monkeypatch):
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1000")
    calls = []
    monkeypatch.setattr(agent, "run_command", lambda argv, **k: calls.append(argv) or "1001")

    assert agent.resubmit_self("/repo/dashboard/agent/dashboard_agent.sbatch") is True
    assert calls[0] == [
        "sbatch",
        "--parsable",
        "--dependency=afterany:1000",
        "/repo/dashboard/agent/dashboard_agent.sbatch",
    ]


def test_a_second_attempt_does_not_queue_a_duplicate(tmp_path, monkeypatch):
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1000")
    calls = []
    monkeypatch.setattr(agent, "run_command", lambda argv, **k: calls.append(argv) or "1001")

    agent.resubmit_self("/repo/x.sbatch")
    agent.resubmit_self("/repo/x.sbatch")
    assert len(calls) == 1


def test_a_stale_lock_from_the_predecessor_does_not_block_the_chain(tmp_path, monkeypatch):
    """The lock records "<predecessor>:<successor>".

    Each generation must ignore a lock written by an earlier one, or the chain
    would stop after exactly one handover.
    """
    _isolate_state(tmp_path, monkeypatch)
    (tmp_path / "resubmit.lock").write_text("1000:1001")

    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    calls = []
    monkeypatch.setattr(agent, "run_command", lambda argv, **k: calls.append(argv) or "1002")

    assert agent.resubmit_self("/repo/x.sbatch") is True
    assert calls[0][2] == "--dependency=afterany:1001"
    assert (tmp_path / "resubmit.lock").read_text() == "1001:1002"


def test_failed_submission_is_reported_so_the_caller_can_retry(tmp_path, monkeypatch):
    """sbatch can fail transiently (a submit limit, a busy controller).

    The agent must learn that so it keeps its remaining wall time and tries
    again, instead of exiting and ending the chain.
    """
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1000")
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: None)

    assert agent.resubmit_self("/repo/x.sbatch") is False
    assert not os.path.exists(str(tmp_path / "resubmit.lock"))


def test_outside_slurm_there_is_nothing_to_chain(tmp_path, monkeypatch):
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: "9999")
    assert agent.resubmit_self("/repo/x.sbatch") is False
