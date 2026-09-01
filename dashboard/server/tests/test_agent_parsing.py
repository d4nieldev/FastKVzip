"""Parsing tests for the cluster agent, against real SLURM output shapes."""

import base64
import gzip

import probe_agent as agent


def test_parse_duration_handles_every_slurm_shape():
    assert agent.parse_duration("01:00:00") == 3600
    assert agent.parse_duration("7-00:00:00") == 604800
    assert agent.parse_duration("12:34") == 754
    assert agent.parse_duration("1-02:03:04") == 93784
    assert agent.parse_duration("00:00:12.345") == 12
    assert agent.parse_duration("UNLIMITED") is None
    assert agent.parse_duration("Partition_Limit") is None
    assert agent.parse_duration("") is None
    assert agent.parse_duration(None) is None


def test_parse_timestamp_rejects_slurm_null_spellings():
    assert agent.parse_timestamp("Unknown") is None
    assert agent.parse_timestamp("N/A") is None
    assert agent.parse_timestamp("None") is None
    assert agent.parse_timestamp("2026-08-29T10:00:00") is not None


def test_parse_timestamp_is_local_time_normalized_to_utc():
    """SLURM prints local cluster time with no zone; the epoch must reflect that."""
    import datetime

    stamp = agent.parse_timestamp("2026-08-29T10:00:00")
    naive = datetime.datetime(2026, 8, 29, 10, 0, 0)
    assert stamp == int(naive.astimezone().timestamp())


def test_parse_exit_code_drops_success():
    assert agent.parse_exit_code("0:0") is None
    assert agent.parse_exit_code("1:0") == "1:0"
    assert agent.parse_exit_code("0:9") == "0:9"


def test_tres_and_gres_extraction():
    tres = "cpu=8,mem=60G,node=1,billing=8,gres/gpu=1,gres/gpu:rtx_pro_6000=1"
    assert agent.tres_field(tres, "mem") == "60G"
    assert agent.tres_field(tres, "cpu") == "8"
    assert agent.tres_field(tres, "nope") is None
    assert agent.gres_from_tres(tres) == "rtx_pro_6000:1"
    assert agent.gres_from_tres("cpu=1,mem=1G") is None


def test_scontrol_line_keeps_values_containing_spaces():
    """Reason and Command hold spaces, so whitespace splitting would corrupt them."""
    line = (
        "JobId=123 JobName=e124-g-rand UserId=danieloh(1001) JobState=PENDING "
        "Reason=Dependency Dependency=afterok:122 Partition=main "
        "Command=/home/danieloh/FastKVzip-implicit/slurm/train_graph.sbatch run name"
    )
    fields = agent.parse_scontrol_line(line)
    assert fields["JobName"] == "e124-g-rand"
    assert fields["Reason"] == "Dependency"
    assert fields["Dependency"] == "afterok:122"
    assert fields["Command"].endswith("train_graph.sbatch run name")


def test_collect_scontrol_jobs_filters_users_and_computes_remaining(monkeypatch):
    output = "\n".join(
        [
            "JobId=1001 JobName=graph-train UserId=danieloh(1001) JobState=RUNNING "
            "Partition=main TimeLimit=01:00:00 RunTime=00:20:00 "
            "SubmitTime=2026-08-29T09:00:00 StartTime=2026-08-29T09:05:00 "
            "EndTime=2026-08-29T10:05:00 NumCPUs=8 NumNodes=1 NodeList=gpu01 "
            "ReqTRES=cpu=8,mem=60G,gres/gpu:rtx_pro_6000=1 "
            "AllocTRES=cpu=8,mem=60G,gres/gpu:rtx_pro_6000=1 Reason=None "
            "WorkDir=/home/danieloh/FastKVzip-implicit "
            "StdOut=/home/danieloh/FastKVzip-implicit/.slurm/logs/1001-graph-train.log",
            "JobId=2002 JobName=someone-else UserId=otheruser(2002) JobState=RUNNING "
            "Partition=main TimeLimit=01:00:00 RunTime=00:01:00",
        ]
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: output)

    jobs = agent.collect_scontrol_jobs("danieloh")
    assert len(jobs) == 1

    job = jobs[0]
    assert job["job_id"] == "1001"
    assert job["state"] == "RUNNING"
    assert job["time_limit_s"] == 3600
    assert job["elapsed_s"] == 1200
    assert job["remaining_s"] == 2400
    assert job["gres"] == "rtx_pro_6000:1"
    assert job["mem_req"] == "60G"
    assert job["std_out"].endswith("1001-graph-train.log")
    # Still running: an end time must not be reported, or window queries and the
    # UI would treat it as finished.
    assert job["end_ts"] is None


def test_pending_start_time_is_an_estimate_not_a_start(monkeypatch):
    """SLURM reports a predicted StartTime for queued jobs.

    Stored as a real start it lands in the future, which drops the job out of
    time-window queries and gives it a wall clock it has not begun to consume.
    """
    output = (
        "JobId=20687878 JobName=fastkvzip-equivalence UserId=danieloh(1) "
        "JobState=PENDING Partition=gpu TimeLimit=06:00:00 RunTime=00:00:00 "
        "SubmitTime=2026-08-29T10:00:00 StartTime=2026-08-30T04:00:01 EndTime=Unknown "
        "NumCPUs=4 NumNodes=1 NodeList=(null) Reason=QOSMaxGRESPerUser "
        "ReqTRES=cpu=1,mem=60G,gres/gpu:rtx_pro_6000=1 "
        "WorkDir=/home/danieloh/FastKVzip-implicit"
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: output)

    job = agent.collect_scontrol_jobs("danieloh")[0]
    assert job["state"] == "PENDING"
    assert job["start_ts"] is None
    assert job["est_start_ts"] == agent.parse_timestamp("2026-08-30T04:00:01")
    assert job["reason"] == "QOSMaxGRESPerUser"


def test_running_start_time_is_a_real_start(monkeypatch):
    output = (
        "JobId=20687850 JobName=qwen3-8b-e1 UserId=danieloh(1) JobState=RUNNING "
        "Partition=gpu TimeLimit=2-00:00:00 RunTime=09:13:47 "
        "SubmitTime=2026-08-29T10:00:00 StartTime=2026-08-29T10:02:16 EndTime=Unknown "
        "NumCPUs=6 NumNodes=1 NodeList=cs-6000-02 "
        "AllocTRES=cpu=6,mem=60G,node=1,billing=122926,gres/gpu=1,gres/gpu:rtx_6000=1 "
        "WorkDir=/home/danieloh/FastKVzip-implicit"
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: output)

    job = agent.collect_scontrol_jobs("danieloh")[0]
    assert job["start_ts"] is not None
    assert job["est_start_ts"] is None
    # 2 days requested, 9h13m47s used.
    assert job["time_limit_s"] == 172800
    assert job["remaining_s"] == 172800 - 33227
    assert job["gres"] == "rtx_6000:1"


def test_unassigned_node_list_is_treated_as_empty(monkeypatch):
    """sacct writes "None assigned" for a job that never got an allocation."""
    row = (
        "20281869|gkv-mask2|FAILED|1:0|2026-08-29T09:00:00|Unknown|"
        "2026-08-29T09:00:00|00:00:00|00:15:00|cpu=6,mem=64G|||gpu|0|1|"
        "None assigned|/home/danieloh/graph-kv"
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: row)
    job = agent.collect_sacct_jobs("danieloh", 30)[0]
    assert job["node_list"] is None
    assert job["start_ts"] is None


def test_collect_scontrol_uses_array_task_ids(monkeypatch):
    output = (
        "JobId=9001 ArrayJobId=9000 ArrayTaskId=3 JobName=arr UserId=danieloh(1) "
        "JobState=RUNNING Partition=main TimeLimit=01:00:00 RunTime=00:00:30"
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: output)
    assert agent.collect_scontrol_jobs("danieloh")[0]["job_id"] == "9000_3"


def test_collect_sacct_folds_steps_into_parent(monkeypatch):
    rows = [
        "1001|graph-train|COMPLETED|0:0|2026-08-29T09:00:00|2026-08-29T09:05:00|"
        "2026-08-29T09:35:00|00:30:00|01:00:00|cpu=8,mem=60G|cpu=8,mem=60G||main|8|1|gpu01|/home/danieloh",
        "1001.batch|batch|COMPLETED|0:0|2026-08-29T09:05:00|2026-08-29T09:05:00|"
        "2026-08-29T09:35:00|00:30:00||||12500000K|main|8|1|gpu01|/home/danieloh",
        "1002|graph-eval|FAILED|1:0|2026-08-29T09:00:00|2026-08-29T09:05:00|"
        "2026-08-29T09:06:00|00:01:00|01:00:00|cpu=8,mem=60G|cpu=8,mem=60G||main|8|1|gpu02|/home/danieloh",
    ]
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: "\n".join(rows))

    jobs = {job["job_id"]: job for job in agent.collect_sacct_jobs("danieloh", 30)}
    assert set(jobs) == {"1001", "1002"}
    assert jobs["1001"]["max_rss"] == "12500000K"
    assert jobs["1001"]["exit_code"] is None
    assert jobs["1002"]["exit_code"] == "1:0"
    assert jobs["1002"]["end_ts"] is not None


def test_collect_sacct_strips_cancelled_by_suffix(monkeypatch):
    row = (
        "1003|x|CANCELLED by 1001|0:0|2026-08-29T09:00:00|2026-08-29T09:05:00|"
        "2026-08-29T09:06:00|00:01:00|01:00:00|||| main|8|1|gpu01|/home/danieloh"
    )
    monkeypatch.setattr(agent, "run_command", lambda *a, **k: row)
    assert agent.collect_sacct_jobs("danieloh", 30)[0]["state"] == "CANCELLED"


def test_merge_prefers_live_values_but_keeps_sacct_extras():
    live = [{"job_id": "1", "state": "RUNNING", "reason": "None", "max_rss": None}]
    history = [{"job_id": "1", "state": "PENDING", "reason": None, "max_rss": "500K"}]
    merged = agent.merge_jobs(live, history)[0]
    assert merged["state"] == "RUNNING"
    assert merged["max_rss"] == "500K"


def test_read_log_chunk_detects_rewind(tmp_path):
    path = tmp_path / "job.log"
    path.write_bytes(b"hello")
    data, offset, rewound = agent.read_log_chunk(str(path), 100)
    assert (data, offset, rewound) == (b"hello", 0, True)


def test_read_log_chunk_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "MAX_LOG_CHUNK_BYTES", 10)
    path = tmp_path / "job.log"
    path.write_bytes(b"x" * 100)
    data, offset, _ = agent.read_log_chunk(str(path), 0)
    assert len(data) == 10 and offset == 0


def test_build_log_payloads_advances_only_via_server_ack(tmp_path):
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"first chunk\n")
    job = {
        "job_id": "1001",
        "state": "RUNNING",
        "std_out": str(path),
        "work_dir": str(tmp_path),
    }
    state = {"logs": {}}

    payloads, _ = agent.build_log_payloads([job], state, now=0)
    assert payloads[0]["offset"] == 0
    assert payloads[0]["raw_bytes"] == 12

    # Nothing acked yet, so the same bytes are offered again -- a dropped POST
    # must not silently lose log data.
    assert agent.build_log_payloads([job], state, now=0)[0][0]["offset"] == 0

    agent.apply_server_response({"ack": {"1001": 12}}, state)
    assert agent.build_log_payloads([job], state, now=0)[0] == []

    path.write_bytes(b"first chunk\nsecond\n")
    assert agent.build_log_payloads([job], state, now=0)[0][0]["offset"] == 12


def test_first_sight_of_a_huge_log_starts_near_the_end(tmp_path, monkeypatch):
    """A job discovered mid-run must not replay its whole history.

    These training logs run to hundreds of MB of tqdm output; shipping from
    byte 0 at the per-poll cap would spend hours on stale output before ever
    showing what the job is doing now.
    """
    monkeypatch.setattr(agent, "INITIAL_BACKFILL_BYTES", 1000)
    monkeypatch.setattr(agent, "MAX_LOG_CHUNK_BYTES", 10_000)

    path = tmp_path / "1001-run.log"
    path.write_bytes(b"O" * 50_000 + b"RECENT\n")
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    # Server-relative offsets stay small even though the file is 50 KB in.
    assert chunks[0]["offset"] == 0
    assert offsets["1001"] == 0
    # It replaces rather than appends, and carries only the tail window.
    assert chunks[0]["truncate"] is True
    assert chunks[0]["raw_bytes"] == 1000
    assert state["logs"]["1001"]["base_offset"] == 50_007 - 1000


def test_tail_window_starts_on_a_line_boundary(tmp_path, monkeypatch):
    """Cutting at an arbitrary byte would make the viewer's first line a fragment."""
    monkeypatch.setattr(agent, "INITIAL_BACKFILL_BYTES", 100)
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"".join(b"line %04d padded out a bit\n" % i for i in range(200)))
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, _ = agent.build_log_payloads([job], state, now=0)
    text = gzip.decompress(base64.b64decode(chunks[0]["data"])).decode()
    assert text.startswith("line ")
    assert text.endswith("\n")


def test_tail_alignment_gives_up_on_a_log_with_no_line_breaks(tmp_path, monkeypatch):
    """Skipping to the next newline must not skip the whole window."""
    monkeypatch.setattr(agent, "INITIAL_BACKFILL_BYTES", 100)
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"x" * 5000)
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, _ = agent.build_log_payloads([job], state, now=0)
    assert chunks and chunks[0]["raw_bytes"] == 100


def test_tail_start_then_follows_appends_normally(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "INITIAL_BACKFILL_BYTES", 1000)
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"O" * 50_000)
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, _ = agent.build_log_payloads([job], state, now=0)
    agent.apply_server_response({"ack": {"1001": chunks[0]["raw_bytes"]}}, state)

    with open(path, "ab") as handle:
        handle.write(b"new line\n")
    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    assert chunks[0]["raw_bytes"] == 9
    assert chunks[0]["truncate"] is False
    assert chunks[0]["offset"] == 1000
    assert offsets["1001"] == 1000


def test_reset_reships_a_fresh_tail_not_the_whole_backlog(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "INITIAL_BACKFILL_BYTES", 1000)
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"O" * 50_000)
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, _ = agent.build_log_payloads([job], state, now=0)
    agent.apply_server_response({"ack": {"1001": chunks[0]["raw_bytes"]}}, state)

    agent.apply_server_response({"ack": {"1001": 0}, "reset": ["1001"]}, state)
    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    assert chunks[0]["offset"] == 0
    assert chunks[0]["truncate"] is True
    assert chunks[0]["raw_bytes"] == 1000


def test_quiet_logs_still_declare_their_offset(tmp_path):
    """A job with no new output must still report where it stands.

    Otherwise a server that lost the log has nothing to notice the gap from,
    and the log would stay missing until the job wrote again -- which for a
    finished job is never.
    """
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"all done\n")
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    agent.build_log_payloads([job], state, now=0)
    agent.apply_server_response({"ack": {"1001": 9}}, state)

    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    assert chunks == []
    assert offsets == {"1001": 9}


def test_head_fingerprint_waits_for_a_full_window(tmp_path):
    """A log still growing through its first kilobytes changes its own head on
    every append; fingerprinting it then would look like a rewrite."""
    path = tmp_path / "small.log"
    path.write_bytes(b"x" * 100)
    assert agent.head_fingerprint(str(path)) is None

    path.write_bytes(b"x" * agent.HEAD_FINGERPRINT_BYTES)
    stable = agent.head_fingerprint(str(path))
    assert stable is not None

    # Appending past the window must not change the fingerprint.
    with open(path, "ab") as handle:
        handle.write(b"y" * 500)
    assert agent.head_fingerprint(str(path)) == stable

    path.write_bytes(b"z" * agent.HEAD_FINGERPRINT_BYTES)
    assert agent.head_fingerprint(str(path)) != stable


def test_log_rewritten_longer_in_place_is_detected(tmp_path):
    """The case shrink and inode checks both miss: same path, same inode,
    different content, longer than what was already shipped."""
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"first attempt\n" + b"a" * agent.HEAD_FINGERPRINT_BYTES)
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    chunks, _ = agent.build_log_payloads([job], state, now=0)
    agent.apply_server_response({"ack": {"1001": chunks[0]["raw_bytes"]}}, state)

    # Rewritten in place with different and longer content.
    path.write_bytes(b"second attempt\n" + b"b" * (agent.HEAD_FINGERPRINT_BYTES * 2))
    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    assert chunks[0]["offset"] == 0
    assert chunks[0]["truncate"] is True
    assert offsets["1001"] == 0


def test_rewound_log_is_flagged_for_truncation(tmp_path):
    """A requeued job overwrites its log; the server must replace, not append."""
    path = tmp_path / "1001-run.log"
    path.write_bytes(b"first attempt, quite long\n")
    job = {"job_id": "1001", "state": "RUNNING", "std_out": str(path), "work_dir": str(tmp_path)}
    state = {"logs": {}}

    # The first chunk of a newly discovered log always replaces: its window is
    # not guaranteed to line up with whatever the server already holds.
    chunks, _ = agent.build_log_payloads([job], state, now=0)
    assert chunks[0]["truncate"] is True
    agent.apply_server_response({"ack": {"1001": 26}}, state)

    path.write_bytes(b"retry\n")
    chunks, offsets = agent.build_log_payloads([job], state, now=0)
    assert chunks[0]["offset"] == 0
    assert chunks[0]["truncate"] is True
    assert offsets == {"1001": 0}


def test_apply_server_response_ack_advances_the_cluster_position(tmp_path):
    path = tmp_path / "run.log"
    path.write_bytes(b"x" * 100)
    state = {"logs": {"1001": {"path": str(path), "base_offset": 40, "file_pos": 40}}}

    # The ack is a server-side offset, so it resumes at base + ack in the file.
    agent.apply_server_response({"ack": {"1001": 60}}, state)
    assert state["logs"]["1001"]["file_pos"] == 100


def test_log_path_falls_back_to_repo_convention(tmp_path):
    logs = tmp_path / ".slurm" / "logs"
    logs.mkdir(parents=True)
    (logs / "1001-e124-g-rand.log").write_text("x")
    job = {"job_id": "1001", "work_dir": str(tmp_path), "std_out": None}
    assert agent.resolve_log_path(job, None).endswith("1001-e124-g-rand.log")


def test_finished_jobs_stop_being_tailed_after_the_grace_window():
    now = 1_000_000
    assert agent.should_tail({"state": "RUNNING", "end_ts": None}, now)
    assert agent.should_tail({"state": "COMPLETED", "end_ts": now - 10}, now)
    assert not agent.should_tail(
        {"state": "COMPLETED", "end_ts": now - agent.TAIL_GRACE_SECONDS - 1}, now
    )


# --------------------------------------------------------------------------- #
# sres collection
# --------------------------------------------------------------------------- #


def test_sres_uses_the_first_shell_that_produces_output(monkeypatch):
    tried = []

    def fake(argv, timeout=60):
        tried.append(argv)
        # An alias resolves only in the interactive shell, so the plain login
        # shell is the one that fails on a real BGU node.
        if argv == ["bash", "-lic", "sres"]:
            return "PARTITION NODE GPU FREE\nmain gpu-01 rtx_6000 3/4\n", None
        return None, "exit 127: bash: sres: command not found"

    monkeypatch.setattr(agent, "run_command_detail", fake)
    assert "rtx_6000" in agent.collect_sres()
    assert tried == [["bash", "-lic", "sres"]]


def test_sres_falls_through_to_the_later_shells(monkeypatch):
    def fake(argv, timeout=60):
        if argv == ["bash", "-lc", "sres"]:
            return "NODE GPU FREE\ngpu-01 rtx_6000 1/4\n", None
        return None, "exit 127"

    monkeypatch.setattr(agent, "run_command_detail", fake)
    assert "rtx_6000" in agent.collect_sres()


def test_sres_reports_why_it_failed_instead_of_returning_nothing(monkeypatch):
    # None stores nothing and renders as no panel at all, which is
    # indistinguishable from a snapshot that has simply not arrived yet.
    monkeypatch.setattr(
        agent,
        "run_command_detail",
        lambda argv, timeout=60: (None, "exit 127: bash: sres: command not found"),
    )
    body = agent.collect_sres()
    assert body is not None
    assert "command not found" in body
    assert "bash -lic sres" in body


def test_sres_treats_an_empty_success_as_a_failure(monkeypatch):
    monkeypatch.setattr(agent, "run_command_detail", lambda argv, timeout=60: ("  \n", None))
    assert "printed nothing" in agent.collect_sres()


# --------------------------------------------------------------------------- #
# Batch script and submission environment
# --------------------------------------------------------------------------- #


def test_batch_script_prefers_the_controllers_own_copy(monkeypatch, tmp_path):
    def fake(argv, timeout=60):
        if argv[:3] == ["scontrol", "write", "batch_script"]:
            with open(argv[4], "w", encoding="utf-8") as handle:
                handle.write("#!/bin/bash\n#SBATCH --gpus=rtx_pro_6000:1\n")
            return "", None
        raise AssertionError("sacct should not be reached when scontrol answers")

    monkeypatch.setattr(agent, "run_command_detail", fake)
    body, source = agent.collect_batch_script("1001", None)
    assert "--gpus=rtx_pro_6000:1" in body
    assert source == "scontrol"


def test_batch_script_falls_back_to_accounting(monkeypatch):
    # The controller forgets a job MinJobAge after it ends; accounting keeps it
    # when the site enabled AccountingStoreFlags=job_script.
    def fake(argv, timeout=60):
        if argv[0] == "scontrol":
            return None, "exit 1: Invalid job id specified"
        return "#!/bin/bash\necho from-accounting\n", None

    monkeypatch.setattr(agent, "run_command_detail", fake)
    body, source = agent.collect_batch_script("1001", None)
    assert "from-accounting" in body
    assert source == "sacct"


def test_batch_script_falls_back_to_the_file_on_disk(monkeypatch, tmp_path):
    script = tmp_path / "train_graph.sbatch"
    script.write_text("#!/bin/bash\necho on-disk\n", encoding="utf-8")
    monkeypatch.setattr(agent, "run_command_detail", lambda argv, timeout=60: (None, "exit 1"))

    body, source = agent.collect_batch_script("1001", str(script))
    # Labelled "disk", never "scontrol": the repository may have moved on since
    # this job was submitted, so it is not necessarily what ran.
    assert "on-disk" in body
    assert source == "disk"


def test_batch_script_reports_unavailable_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(agent, "run_command_detail", lambda argv, timeout=60: (None, "exit 1"))
    assert agent.collect_batch_script("1001", "/no/such/file") == (None, "unavailable")
    assert agent.collect_job_env("1001") == (None, "unavailable")


def test_a_job_is_only_asked_about_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent, "run_command_detail", lambda argv, timeout=60: (calls.append(argv), (None, "x"))[1]
    )
    state = {}
    jobs = [{"job_id": "1001"}]
    first = agent.build_script_payloads(jobs, state)
    before = len(calls)
    second = agent.build_script_payloads(jobs, state)

    assert len(first) == 1 and second == []
    # A failure is remembered too, or every poll would re-run scontrol forever.
    assert first[0]["script_source"] == "unavailable"
    assert first[0]["note"] and "AccountingStoreFlags" in first[0]["note"]
    assert len(calls) == before


def test_script_state_is_pruned_with_the_job(monkeypatch):
    state = {"logs": {"1001": {}}, "scripts": {"1001": True, "1002": True}}
    agent.prune_state(state, {"1001"})
    assert state["scripts"] == {"1001": True}
