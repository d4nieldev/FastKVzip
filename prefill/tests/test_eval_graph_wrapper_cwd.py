import os
import subprocess
from pathlib import Path


def test_standard_wrapper_uses_allocated_job_cwd_not_the_submission_shell(tmp_path):
    project = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then echo 4.0.0; '
        'elif [ "$1" = "--version" ]; then echo Python-fixture; '
        'else printf "%s\\n" "$@"; fi\n'
    )
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\necho GPU-fixture\n")
    for executable in (python, nvidia):
        executable.chmod(0o755)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("")
    submission_dir = tmp_path / "unrelated-submission-directory"
    submission_dir.mkdir()

    completed = subprocess.run(
        [
            "bash",
            str(project / "slurm/eval_graph.sbatch"),
            "cwd-check",
            "--data",
            "scbench_kv",
        ],
        cwd=project,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FASTKVZIP_VENV": str(venv),
            "SLURM_SUBMIT_DIR": str(submission_dir),
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NAME": "cwd-check",
            "EVAL_GRAPH_SCRIPT": "prefill/eval_graph.py",
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"results={project}/results/cwd-check\n" in completed.stdout
    assert f"--run-dir\n{project}/results/cwd-check\n" in completed.stdout
    assert "not a git repository" not in completed.stderr
