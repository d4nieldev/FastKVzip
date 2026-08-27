#!/usr/bin/env bash
# One-example evaluation pilot defaults. Override time, memory, and GPU type at
# sbatch time only after checking live cluster limits.
#SBATCH --job-name=graph-eval
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --output=.slurm/logs/graph-eval-%j.out

set -euo pipefail

readonly RUN_NAME="${1:?usage: sbatch ... eval_graph.sbatch RUN_NAME [eval_graph.py arguments]}"
shift
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [[ -z "${FASTKVZIP_VENV:-}" || ! -f "${FASTKVZIP_VENV:-}/bin/activate" ]]; then
    echo "export FASTKVZIP_VENV=/path/to/fastkvzip before submission" >&2
    exit 2
fi

readonly PROJECT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
source "$FASTKVZIP_VENV/bin/activate"
cd "$PROJECT_DIR/prefill"

echo "commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python -u eval_graph.py "$@" --tag "$RUN_NAME"
