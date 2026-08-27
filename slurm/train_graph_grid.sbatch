#!/usr/bin/env bash
# submit_graph_grid.sh supplies measured time, memory, and GPU values.
#SBATCH --job-name=graph-grid
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --array=0-8%1
#SBATCH --output=.slurm/logs/graph-grid-%A_%a.out

set -euo pipefail

readonly PROJECT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

case "$TASK_ID" in
    0) EPOCHS=1; GATE=random; FROZEN=false ;;
    1) EPOCHS=1; GATE=pretrained; FROZEN=false ;;
    2) EPOCHS=1; GATE=pretrained; FROZEN=true ;;
    3) EPOCHS=2; GATE=random; FROZEN=false ;;
    4) EPOCHS=2; GATE=pretrained; FROZEN=false ;;
    5) EPOCHS=2; GATE=pretrained; FROZEN=true ;;
    6) EPOCHS=4; GATE=random; FROZEN=false ;;
    7) EPOCHS=4; GATE=pretrained; FROZEN=false ;;
    8) EPOCHS=4; GATE=pretrained; FROZEN=true ;;
    *) echo "unknown grid task: $TASK_ID" >&2; exit 2 ;;
esac

readonly TRAINING_STATE="$( [[ "$FROZEN" == true ]] && printf frozen || printf trainable )"
readonly RUN_NAME="qwen3-8b-e${EPOCHS}-gate-${GATE}-${TRAINING_STATE}-seed0"

export MODEL_ID="Qwen/Qwen3-8B"
export FASTKVZIP_VENV="${FASTKVZIP_VENV:-/home/danieloh/.venvs/fastkvzip}"
export OUTPUT_ROOT="$PROJECT_DIR/graph_checkpoints"
export TEACHER_CACHE_DIR="/groups/ydar_group/danieloh/fastkvzip-implicit/teacher-cache-qwen3-8b-prefill16k"

if [[ -e "$OUTPUT_ROOT/$RUN_NAME" ]]; then
    echo "refusing to reuse existing output directory: $OUTPUT_ROOT/$RUN_NAME" >&2
    exit 2
fi

ARGS=(
    --epochs "$EPOCHS"
    --seed 0
    --save-strategy epochs --save-every 1
    --eval-strategy epochs --eval-every 1
    --no-save-best
)
if [[ "$GATE" == random ]]; then
    ARGS+=(--no-freeze-gate)
elif [[ "$FROZEN" == true ]]; then
    ARGS+=(--gate-checkpoint fastkvzip --freeze-gate)
else
    ARGS+=(--gate-checkpoint fastkvzip --no-freeze-gate)
fi

echo "grid_task=$TASK_ID run_name=$RUN_NAME epochs=$EPOCHS gate=$GATE frozen=$FROZEN"
exec bash "$PROJECT_DIR/slurm/train_graph.sbatch" "$RUN_NAME" "${ARGS[@]}"
