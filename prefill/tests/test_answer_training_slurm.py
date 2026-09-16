import os
import subprocess
from pathlib import Path

import pytest


def _environment(tmp_path):
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "activate").write_text("", encoding="utf-8")
    return {
        **os.environ,
        "FASTKVZIP_VENV": str(environment),
        "OUTPUT_ROOT": "/durable/graph-answer-checkpoints",
        # The helper must not need sbatch to print a dry-run command.
        "PATH": "/usr/bin:/bin",
    }


def _run(helper, environment, *arguments):
    return subprocess.run(
        ["bash", helper, *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )


RESOURCES = (
    ("--gpu", "rtx_6000:1", "--gpus"),
    ("--time", "01:00:00", "--time"),
    ("--mem", "60G", "--mem"),
    ("--tmp", "40G", "--tmp"),
)


def _resource_arguments(*, except_resource=None):
    arguments = ["answer-run"]
    for resource, value, _ in RESOURCES:
        if resource != except_resource:
            arguments.extend((resource, value))
    return arguments


@pytest.mark.parametrize("shuffle_flag", ("--shuffle-data", "--no-shuffle-data"))
def test_answer_training_submit_helper_dry_run_forwards_durable_arguments(tmp_path, shuffle_flag):
    project = Path(__file__).resolve().parents[2]
    helper = project / "slurm" / "submit_train_graph_answer.sh"
    batch = project / "slurm" / "train_graph_answer.sbatch"
    subprocess.run(["bash", "-n", helper, batch], check=True)

    completed = _run(
        helper,
        _environment(tmp_path),
        "answer-resume",
        "--gpu=rtx_6000:1",
        "--time",
        "01:00:00",
        "--mem=60G",
        "--tmp",
        "40G",
        "--resume",
        "/durable/graph-answer-checkpoints/answer-resume/last.pt",
        "--answer-cache-dir",
        "/durable/answer-cache",
        "--gradient-accumulation-steps",
        "8",
        shuffle_flag,
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    command = completed.stdout
    assert "--gpus=rtx_6000:1" in command
    assert "--time=01:00:00" in command
    assert "--mem=60G" in command
    assert "--tmp=40G" in command
    assert "--job-name=answer-resume" in command
    assert str(batch).replace(" ", r"\ ") in command
    assert "--resume /durable/graph-answer-checkpoints/answer-resume/last.pt" in command
    assert "--answer-cache-dir /durable/answer-cache" in command
    assert "--gradient-accumulation-steps 8" in command
    assert shuffle_flag in command
    assert "OUTPUT_ROOT=/durable/graph-answer-checkpoints" in command
    assert "dependency" not in command
    assert "warm" not in command


@pytest.mark.parametrize(("resource", "value", "sbatch_option"), RESOURCES)
@pytest.mark.parametrize("form", ("split", "equals"))
def test_answer_training_submit_helper_accepts_every_resource_form(
    tmp_path, resource, value, sbatch_option, form
):
    project = Path(__file__).resolve().parents[2]
    helper = project / "slurm" / "submit_train_graph_answer.sh"
    arguments = _resource_arguments(except_resource=resource)
    if form == "split":
        arguments.extend((resource, value))
    else:
        arguments.append(f"{resource}={value}")
    arguments.extend(("--resume", "/durable/last.pt", "--dry-run"))

    completed = _run(helper, _environment(tmp_path), *arguments)

    assert completed.returncode == 0, completed.stderr
    assert f"{sbatch_option}={value}" in completed.stdout


@pytest.mark.parametrize("resource", tuple(resource for resource, _, _ in RESOURCES))
@pytest.mark.parametrize("malformed", ("trailing", "next-option", "empty-equals"))
def test_answer_training_submit_helper_rejects_malformed_resource_values(
    tmp_path, resource, malformed
):
    project = Path(__file__).resolve().parents[2]
    helper = project / "slurm" / "submit_train_graph_answer.sh"
    arguments = _resource_arguments(except_resource=resource)
    if malformed == "trailing":
        arguments.append(resource)
    elif malformed == "next-option":
        next_resource = "--time" if resource != "--time" else "--gpu"
        next_value = dict((option, value) for option, value, _ in RESOURCES)[next_resource]
        arguments[1:1] = (resource, next_resource, next_value)
    else:
        arguments.append(f"{resource}=")
    arguments.extend(("--resume", "/durable/last.pt", "--dry-run"))

    completed = _run(helper, _environment(tmp_path), *arguments)

    assert completed.returncode == 2
    assert f"missing value for {resource}" in completed.stderr


@pytest.mark.parametrize("output_option", ("--output-dir", "--output-dir=/other"))
def test_answer_training_submit_helper_owns_output_directory(tmp_path, output_option):
    project = Path(__file__).resolve().parents[2]
    helper = project / "slurm" / "submit_train_graph_answer.sh"
    arguments = [
        "answer-run",
        "--gpu",
        "rtx_6000:1",
        "--time",
        "01:00:00",
        "--mem",
        "60G",
        "--tmp",
        "40G",
        "--resume",
        "/durable/last.pt",
    ]
    if output_option == "--output-dir":
        arguments.extend((output_option, "/other"))
    else:
        arguments.append(output_option)

    completed = _run(helper, _environment(tmp_path), *arguments)

    assert completed.returncode == 2
    assert "helper owns --output-dir" in completed.stderr


def _batch_environment(tmp_path, *, run_name="answer-run", fail_training=False):
    """Run the sbatch body with stub python/nvidia-smi so no GPU or venv is needed."""

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    calls = tmp_path / "python-calls.txt"
    exit_code = 1 if fail_training else 0
    (stubs / "python").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *--version* ]]; then echo "Python 3.12.0"; exit 0; fi\n'
        f'printf "%s\\n" "$*" >> {calls}\n'
        f'if [[ "$*" == *train_graph_answer.py* ]]; then exit {exit_code}; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (stubs / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for stub in ("python", "nvidia-smi"):
        (stubs / stub).chmod(0o755)

    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "activate").write_text("", encoding="utf-8")

    project = Path(__file__).resolve().parents[2]
    return calls, {
        **os.environ,
        "PATH": f"{stubs}:/usr/bin:/bin",
        "FASTKVZIP_VENV": str(environment),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "SLURM_SUBMIT_DIR": str(project),
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_NAME": run_name,
    }


def _run_batch(environment, *arguments):
    project = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["bash", str(project / "slurm" / "train_graph_answer.sbatch"), *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )


def test_training_alone_runs_no_evaluation(tmp_path):
    calls, environment = _batch_environment(tmp_path)

    completed = _run_batch(environment, "answer-run", "--loss", "kl")

    assert completed.returncode == 0, completed.stderr
    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "train_graph_answer.py" in invocations[0]


def test_post_training_evaluation_runs_in_the_same_job_after_a_successful_run(tmp_path):
    calls, environment = _batch_environment(tmp_path)
    run_dir = tmp_path / "runs" / "answer-run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_text("", encoding="utf-8")
    environment["POST_TRAIN_EVAL_ARGS"] = "-d scbench_qa_eng --num 50"

    completed = _run_batch(environment, "answer-run", "--loss", "kl")

    assert completed.returncode == 0, completed.stderr
    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    assert "train_graph_answer.py" in invocations[0]
    evaluation = invocations[1]
    assert "eval_chunk.py" in evaluation
    assert f"-g {run_dir / 'best.pt'}" in evaluation
    assert "-d scbench_qa_eng --num 50" in evaluation
    assert "--existing-results resume" in evaluation


def test_failed_training_never_reaches_evaluation(tmp_path):
    calls, environment = _batch_environment(tmp_path, fail_training=True)
    run_dir = tmp_path / "runs" / "answer-run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_text("", encoding="utf-8")
    environment["POST_TRAIN_EVAL_ARGS"] = "-d scbench_qa_eng"

    completed = _run_batch(environment, "answer-run", "--loss", "kl")

    assert completed.returncode != 0
    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "eval_chunk.py" not in "\n".join(invocations)


def test_evaluation_falls_back_to_last_when_best_was_not_saved(tmp_path):
    calls, environment = _batch_environment(tmp_path)
    run_dir = tmp_path / "runs" / "answer-run"
    run_dir.mkdir(parents=True)
    (run_dir / "last.pt").write_text("", encoding="utf-8")
    environment["POST_TRAIN_EVAL_ARGS"] = "-d scbench_qa_eng"

    completed = _run_batch(environment, "answer-run", "--no-save-best")

    assert completed.returncode == 0, completed.stderr
    assert f"-g {run_dir / 'last.pt'}" in calls.read_text(encoding="utf-8")


def test_scratch_directory_is_derived_when_the_node_does_not_export_it(tmp_path):
    """cs-* and ise-6000-* nodes create /scratch/USER/JOBID without exporting it."""

    calls, environment = _batch_environment(tmp_path)
    scratch = Path("/scratch") / os.environ.get("USER", "") / "12345"
    del environment["SLURM_TMPDIR"]

    completed = _run_batch(environment, "answer-run", "--loss", "kl")

    if scratch.is_dir():
        assert completed.returncode == 0, completed.stderr
        assert calls.read_text(encoding="utf-8")
    else:
        # No such directory here, so the guard must still refuse rather than
        # silently writing the Hugging Face cache into the home directory.
        assert completed.returncode == 2
        assert "SLURM_TMPDIR is required" in completed.stderr
