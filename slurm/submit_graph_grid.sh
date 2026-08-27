#!/usr/bin/env bash
# Submit the nine valid Qwen3-8B graph-training configurations.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash slurm/submit_graph_grid.sh --gpu TYPE:1 --time D-HH:MM:SS --mem SIZE [options]

Required after running sres:
  --gpu TYPE:1       rtx_pro_6000:1 when available, otherwise rtx_6000:1
  --time VALUE       measured full-run time request
  --mem VALUE        measured full-run memory request

Options:
  --max-parallel N   concurrent dependent array tasks, 1 through 8 (default: 2)
  --dry-run          print the two sbatch commands without submitting
EOF
}

GPU=""
TIME=""
MEM=""
MAX_PARALLEL=2
DRY_RUN=false
while (( $# )); do
    case "$1" in
        --gpu|--time|--mem|--max-parallel)
            if (( $# < 2 )); then
                echo "missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --gpu) GPU="$2" ;;
                --time) TIME="$2" ;;
                --mem) MEM="$2" ;;
                --max-parallel) MAX_PARALLEL="$2" ;;
            esac
            shift 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$GPU" || -z "$TIME" || -z "$MEM" ]]; then
    usage >&2
    exit 2
fi
if ! [[ "$MAX_PARALLEL" =~ ^[1-8]$ ]]; then
    echo "--max-parallel must be an integer from 1 through 8" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
readonly BATCH_SCRIPT="$PROJECT_DIR/slurm/train_graph_grid.sbatch"
readonly CACHE_DIR="/groups/ydar_group/danieloh/fastkvzip-implicit/teacher-cache-qwen3-8b-prefill16k"

cd "$PROJECT_DIR"
COMMON=(--gpus="$GPU" --time="$TIME" --mem="$MEM")

if [[ "$DRY_RUN" == true ]]; then
    printf 'warm: sbatch %q %q %q --array=0-0 %q\n' "${COMMON[0]}" "${COMMON[1]}" "${COMMON[2]}" "$BATCH_SCRIPT"
    printf 'grid: sbatch %q %q %q --dependency=afterok:WARM_JOB_ID --array=1-8%%%s %q\n' "${COMMON[0]}" "${COMMON[1]}" "${COMMON[2]}" "$MAX_PARALLEL" "$BATCH_SCRIPT"
    printf 'cache: %s\n' "$CACHE_DIR"
    exit 0
fi

mkdir -p "$PROJECT_DIR/.slurm/logs"
readonly WARM_SUBMISSION="$(sbatch --parsable "${COMMON[@]}" --array=0-0 "$BATCH_SCRIPT")"
readonly WARM_JOB_ID="${WARM_SUBMISSION%%;*}"
if ! [[ "$WARM_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "could not parse warm-up job ID: $WARM_SUBMISSION" >&2
    exit 1
fi
readonly GRID_SUBMISSION="$(sbatch --parsable "${COMMON[@]}" --dependency="afterok:$WARM_JOB_ID" --array="1-8%$MAX_PARALLEL" "$BATCH_SCRIPT")"

printf 'cache warm-up job: %s\n' "$WARM_JOB_ID"
printf 'dependent grid array: %s\n' "$GRID_SUBMISSION"
printf 'shared teacher cache: %s\n' "$CACHE_DIR"
