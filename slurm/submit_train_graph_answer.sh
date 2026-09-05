#!/usr/bin/env bash
# Submit one answer-supervised graph-training job.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash slurm/submit_train_graph_answer.sh RUN_NAME --gpu TYPE:1 --time D-HH:MM:SS --mem SIZE --tmp SIZE [train_graph_answer.py options]

Required after running sres:
  RUN_NAME        unique training name
  --gpu TYPE:1    GPU selected from sres
  --time VALUE    measured training time request
  --mem VALUE     measured training memory request
  --tmp SIZE      Slurm scratch allocation for transient Hugging Face caches

The helper owns --output-dir as OUTPUT_ROOT/RUN_NAME. Set durable OUTPUT_ROOT
before submission; it defaults to graph_checkpoints/answer in this project.
All other options are forwarded unchanged to prefill/train_graph_answer.py.
Use --dry-run to print the sbatch command without submitting it.
EOF
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

readonly RUN_NAME="$1"
shift
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi

GPU=""
TIME=""
MEM=""
TMP=""
DRY_RUN=false
TRAIN_ARGS=()
while (( $# )); do
    case "$1" in
        --gpu|--time|--mem|--tmp)
            if (( $# < 2 )); then
                echo "missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --gpu) GPU="$2" ;;
                --time) TIME="$2" ;;
                --mem) MEM="$2" ;;
                --tmp) TMP="$2" ;;
            esac
            shift 2
            ;;
        --gpu=*) GPU="${1#*=}"; shift ;;
        --time=*) TIME="${1#*=}"; shift ;;
        --mem=*) MEM="${1#*=}"; shift ;;
        --tmp=*) TMP="${1#*=}"; shift ;;
        --output-dir|--output-dir=*)
            echo "the helper owns --output-dir; set OUTPUT_ROOT instead" >&2
            exit 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$GPU" || -z "$TIME" || -z "$MEM" || -z "$TMP" ]]; then
    usage >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
readonly BATCH_SCRIPT="$PROJECT_DIR/slurm/train_graph_answer.sbatch"
readonly LOG_DIR="$PROJECT_DIR/.slurm/logs"
readonly FASTKVZIP_VENV="${FASTKVZIP_VENV:-/home/danieloh/.venvs/fastkvzip}"

if [[ ! -f "$FASTKVZIP_VENV/bin/activate" ]]; then
    echo "FastKVzip environment does not exist: $FASTKVZIP_VENV" >&2
    exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/graph_checkpoints/answer}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$PROJECT_DIR/$OUTPUT_ROOT"
fi
readonly OUTPUT_ROOT

COMMAND=(
    sbatch --parsable
    --job-name="$RUN_NAME"
    --output="$LOG_DIR/%j-%x.log"
    --gpus="$GPU"
    --time="$TIME"
    --mem="$MEM"
    --tmp="$TMP"
    --export="ALL,FASTKVZIP_VENV=$FASTKVZIP_VENV,OUTPUT_ROOT=$OUTPUT_ROOT"
    "$BATCH_SCRIPT" "$RUN_NAME" "${TRAIN_ARGS[@]}"
)

if [[ "$DRY_RUN" == true ]]; then
    printf '  '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
readonly SUBMISSION="$("${COMMAND[@]}")"
readonly JOB_ID="${SUBMISSION%%;*}"
if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "could not parse job ID: $SUBMISSION" >&2
    exit 1
fi

echo "submitted job_id=$JOB_ID name=$RUN_NAME"
echo "log=$LOG_DIR/$JOB_ID-$RUN_NAME.log"
echo "checkpoints=$OUTPUT_ROOT/$RUN_NAME"
