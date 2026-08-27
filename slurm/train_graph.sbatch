#!/usr/bin/env bash
# One-context pilot defaults. Override time, memory, tmp, and GPU type at sbatch time
# only after checking live cluster limits.
#SBATCH --job-name=graph-train
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --tmp=40G
#SBATCH --output=.slurm/logs/graph-train-%j.out

set -euo pipefail

readonly RUN_NAME="${1:?usage: sbatch ... train_graph.sbatch RUN_NAME [train_graph.py arguments]}"
shift
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [[ -z "${FASTKVZIP_VENV:-}" || ! -f "${FASTKVZIP_VENV:-}/bin/activate" ]]; then
    echo "export FASTKVZIP_VENV=/path/to/fastkvzip before submission" >&2
    exit 2
fi
if [[ -z "${MODEL_ID:-}" ]]; then
    echo "export MODEL_ID=... before submission" >&2
    exit 2
fi

readonly PROJECT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/graph}"
readonly RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
readonly SCRATCH_ROOT="${SLURM_TMPDIR:-/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}}"
readonly TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-$SCRATCH_ROOT/teacher-cache}"
mkdir -p "$RUN_DIR" "$TEACHER_CACHE_DIR"

export WANDB_DIR="$RUN_DIR/wandb"
source "$FASTKVZIP_VENV/bin/activate"
cd "$PROJECT_DIR/prefill"

echo "commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
echo "run_dir=$RUN_DIR"
echo "teacher_cache_dir=$TEACHER_CACHE_DIR"
python --version
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python -u train_graph.py "$@" \
    --model "$MODEL_ID" \
    --teacher-cache-dir "$TEACHER_CACHE_DIR" \
    --output-dir "$RUN_DIR" \
    --wandb-name "$RUN_NAME"
