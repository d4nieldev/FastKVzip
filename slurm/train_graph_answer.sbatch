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
# Only the ise-6000p nodes export SLURM_TMPDIR. The cs-* and ise-6000-* nodes
# create /scratch/USER/JOBID and print it as SLURM_SCRATCH_DIR in the prolog
# banner without ever exporting it, so a fresh checkout fails there.
if [[ -z "${SLURM_TMPDIR:-}" && -n "${SLURM_JOB_ID:-}" && -d "/scratch/$USER/$SLURM_JOB_ID" ]]; then
    SLURM_TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
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

# Optional benchmark evaluation in the same allocation. `set -e` above means this
# is reached only when training exited 0, so a failed run never evaluates a stale
# checkpoint; and it reuses the GPU this job already holds rather than queueing a
# second one, which on a busy cluster is where the wall-clock goes.
#
# POST_TRAIN_EVAL_ARGS is forwarded verbatim to the evaluator, so no single
# argument in it may contain whitespace.
if [[ -n "${POST_TRAIN_EVAL_ARGS:-}" ]]; then
    eval_checkpoint="$RUN_DIR/best.pt"
    [[ -f "$eval_checkpoint" ]] || eval_checkpoint="$RUN_DIR/last.pt"
    if [[ ! -f "$eval_checkpoint" ]]; then
        echo "post-training evaluation found no checkpoint under $RUN_DIR" >&2
        exit 2
    fi
    read -ra post_train_eval_args <<<"$POST_TRAIN_EVAL_ARGS"
    echo "post_train_eval_checkpoint=$eval_checkpoint"
    python -u "${POST_TRAIN_EVAL_SCRIPT:-eval_chunk.py}" \
        -g "$eval_checkpoint" \
        "${post_train_eval_args[@]}" \
        --run-dir "$RUN_DIR/eval" \
        --existing-results resume
fi
