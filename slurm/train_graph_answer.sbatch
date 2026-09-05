#!/usr/bin/env bash
# One answer-supervised graph-training run. Resources come from the submit helper.
#SBATCH --job-name=graph-answer-train
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --output=.slurm/logs/%j-%x.log

set -euo pipefail

readonly RUN_NAME="${1:?usage: sbatch --job-name=RUN_NAME ... train_graph_answer.sbatch RUN_NAME [train_graph_answer.py arguments]}"
shift
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" != "$RUN_NAME" ]]; then
    echo "submit with --job-name=$RUN_NAME so Slurm, W&B, and checkpoint names match" >&2
    exit 2
fi
if [[ -z "${FASTKVZIP_VENV:-}" || ! -f "${FASTKVZIP_VENV:-}/bin/activate" ]]; then
    echo "export FASTKVZIP_VENV=/path/to/fastkvzip before submission" >&2
    exit 2
fi
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "SLURM_TMPDIR is required; submit with --tmp SIZE" >&2
    exit 2
fi

readonly PROJECT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/graph_checkpoints/answer}"
readonly RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
readonly HF_CACHE_DIR="$SLURM_TMPDIR/huggingface"
mkdir -p "$RUN_DIR" "$HF_CACHE_DIR"

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export WANDB_DIR="$RUN_DIR/wandb"

source "$FASTKVZIP_VENV/bin/activate"
cd "$PROJECT_DIR/prefill"

echo "commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
echo "output_dir=$RUN_DIR"
echo "hf_cache_dir=$HF_CACHE_DIR"
python --version
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python -u train_graph_answer.py "$@" \
    --output-dir "$RUN_DIR" \
    --wandb-name "$RUN_NAME"
