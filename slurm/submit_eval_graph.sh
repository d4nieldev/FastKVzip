#!/usr/bin/env bash
# Submit one graph evaluation while forwarding all non-resource arguments.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash slurm/submit_eval_graph.sh RUN_NAME --gpu TYPE:1 --time D-HH:MM:SS --mem SIZE --graph-checkpoint PATH [helper and evaluation options]

Required after running sres:
  RUN_NAME                unique evaluation name
  --gpu TYPE:1            rtx_pro_6000:1 when available, otherwise rtx_6000:1
  --time VALUE            measured evaluation time request
  --mem VALUE             measured evaluation memory request
  --graph-checkpoint PATH absolute path or path relative to the project root

All other options are forwarded unchanged to prefill/eval_graph.py.

Result handling:
  --existing-results MODE  fail (default), resume, or overwrite

Optional W&B upload after complete benchmark evaluation:
  --log-to-wandb
  --wandb-project PROJECT  required with --log-to-wandb
  --wandb-entity ENTITY     optional

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
CHECKPOINT_INPUT=""
EXISTING_RESULTS="fail"
LOG_TO_WANDB=false
WANDB_PROJECT=""
WANDB_ENTITY=""
DRY_RUN=false
EVAL_ARGS=()
while (( $# )); do
    case "$1" in
        --gpu|--time|--mem)
            if (( $# < 2 )); then
                echo "missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --gpu) GPU="$2" ;;
                --time) TIME="$2" ;;
                --mem) MEM="$2" ;;
            esac
            shift 2
            ;;
        --gpu=*) GPU="${1#*=}"; shift ;;
        --time=*) TIME="${1#*=}"; shift ;;
        --mem=*) MEM="${1#*=}"; shift ;;
        --graph-checkpoint)
            if (( $# < 2 )); then
                echo "missing value for --graph-checkpoint" >&2
                exit 2
            fi
            if [[ -n "$CHECKPOINT_INPUT" ]]; then
                echo "--graph-checkpoint may be passed only once" >&2
                exit 2
            fi
            CHECKPOINT_INPUT="$2"
            shift 2
            ;;
        --graph-checkpoint=*)
            if [[ -n "$CHECKPOINT_INPUT" ]]; then
                echo "--graph-checkpoint may be passed only once" >&2
                exit 2
            fi
            CHECKPOINT_INPUT="${1#*=}"
            shift
            ;;
        --existing-results)
            if (( $# < 2 )); then
                echo "missing value for --existing-results" >&2
                exit 2
            fi
            EXISTING_RESULTS="$2"
            shift 2
            ;;
        --existing-results=*) EXISTING_RESULTS="${1#*=}"; shift ;;
        --log-to-wandb) LOG_TO_WANDB=true; shift ;;
        --wandb-project)
            if (( $# < 2 )); then
                echo "missing value for --wandb-project" >&2
                exit 2
            fi
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
        --wandb-entity)
            if (( $# < 2 )); then
                echo "missing value for --wandb-entity" >&2
                exit 2
            fi
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb-entity=*) WANDB_ENTITY="${1#*=}"; shift ;;
        --run-dir|--run-dir=*)
            echo "the helper owns --run-dir; use RUN_NAME instead" >&2
            exit 2
            ;;
        --tag|--tag=*)
            echo "the helper owns --tag; use RUN_NAME instead" >&2
            exit 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) EVAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$GPU" || -z "$TIME" || -z "$MEM" || -z "$CHECKPOINT_INPUT" ]]; then
    usage >&2
    exit 2
fi
if [[ "$EXISTING_RESULTS" != "fail" && "$EXISTING_RESULTS" != "resume" && "$EXISTING_RESULTS" != "overwrite" ]]; then
    echo "--existing-results must be fail, resume, or overwrite" >&2
    exit 2
fi
if [[ "$LOG_TO_WANDB" == true && -z "$WANDB_PROJECT" ]]; then
    echo "--wandb-project is required with --log-to-wandb" >&2
    exit 2
fi
if [[ "$LOG_TO_WANDB" == false && ( -n "$WANDB_PROJECT" || -n "$WANDB_ENTITY" ) ]]; then
    echo "--wandb-project and --wandb-entity require --log-to-wandb" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
readonly BATCH_SCRIPT="$PROJECT_DIR/slurm/eval_graph.sbatch"
readonly LOG_DIR="$PROJECT_DIR/.slurm/logs"
readonly RESULTS_ROOT="$PROJECT_DIR/results"
readonly RUN_DIR="$RESULTS_ROOT/$RUN_NAME"
readonly METRICS_PATH="$RUN_DIR/metrics.json"
readonly FASTKVZIP_VENV="${FASTKVZIP_VENV:-/home/danieloh/.venvs/fastkvzip}"

if [[ "$CHECKPOINT_INPUT" = /* ]]; then
    CHECKPOINT_CANDIDATE="$CHECKPOINT_INPUT"
else
    CHECKPOINT_CANDIDATE="$PROJECT_DIR/$CHECKPOINT_INPUT"
fi
if [[ ! -f "$CHECKPOINT_CANDIDATE" ]]; then
    echo "checkpoint does not exist: $CHECKPOINT_CANDIDATE" >&2
    exit 2
fi
readonly CHECKPOINT_DIR="$(cd -- "$(dirname -- "$CHECKPOINT_CANDIDATE")" && pwd -P)"
readonly CHECKPOINT="$CHECKPOINT_DIR/$(basename -- "$CHECKPOINT_CANDIDATE")"
EVAL_ARGS=(--graph-checkpoint "$CHECKPOINT" "${EVAL_ARGS[@]}")

if [[ ! -f "$FASTKVZIP_VENV/bin/activate" ]]; then
    echo "FastKVzip environment does not exist: $FASTKVZIP_VENV" >&2
    exit 2
fi

if [[ "$EXISTING_RESULTS" == "fail" && -e "$RUN_DIR" ]]; then
    echo "evaluation run already exists: $RUN_NAME" >&2
    echo "result: $RUN_DIR" >&2
    exit 2
fi

BATCH_ARGS=(--existing-results "$EXISTING_RESULTS")
if [[ "$LOG_TO_WANDB" == true ]]; then
    BATCH_ARGS+=(--log-to-wandb --wandb-project "$WANDB_PROJECT")
    if [[ -n "$WANDB_ENTITY" ]]; then
        BATCH_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

COMMAND=(
    sbatch --parsable
    --job-name="$RUN_NAME"
    --output="$LOG_DIR/%j-%x.log"
    --gpus="$GPU"
    --time="$TIME"
    --mem="$MEM"
    --export="ALL,FASTKVZIP_VENV=$FASTKVZIP_VENV"
    "$BATCH_SCRIPT" "$RUN_NAME" "${BATCH_ARGS[@]}" -- "${EVAL_ARGS[@]}"
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
echo "results=$RUN_DIR"
echo "metrics=$METRICS_PATH"
