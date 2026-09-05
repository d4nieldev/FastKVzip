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


def test_answer_training_submit_helper_dry_run_forwards_durable_arguments(tmp_path):
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
    assert "OUTPUT_ROOT=/durable/graph-answer-checkpoints" in command
    assert "dependency" not in command
    assert "warm" not in command


@pytest.mark.parametrize("missing", ("--gpu", "--time", "--mem", "--tmp"))
def test_answer_training_submit_helper_requires_each_resource(tmp_path, missing):
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
        "--dry-run",
    ]
    index = arguments.index(missing)
    del arguments[index : index + 2]

    completed = _run(helper, _environment(tmp_path), *arguments)

    assert completed.returncode == 2
    assert "Required" in completed.stderr


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
