#!/usr/bin/env bash
# One-example evaluation pilot defaults. Override time, memory, and GPU type at
# sbatch time only after checking live cluster limits.
#SBATCH --job-name=graph-eval
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --output=.slurm/logs/%j-%x.log

set -euo pipefail

readonly RUN_NAME="${1:?usage: sbatch ... eval_graph.sbatch RUN_NAME [batch options] -- [eval_graph.py arguments]}"
shift
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" != "$RUN_NAME" ]]; then
    echo "submit with --job-name=$RUN_NAME so the job and evaluation names match" >&2
    exit 2
fi
if [[ -z "${FASTKVZIP_VENV:-}" || ! -f "${FASTKVZIP_VENV:-}/bin/activate" ]]; then
    echo "export FASTKVZIP_VENV=/path/to/fastkvzip before submission" >&2
    exit 2
fi

readonly PROJECT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
readonly RESULTS_ROOT="$PROJECT_DIR/results"
readonly RUN_DIR="$RESULTS_ROOT/$RUN_NAME"
readonly METRICS_PATH="$RUN_DIR/metrics.json"

EXISTING_RESULTS="fail"
LOG_TO_WANDB=false
WANDB_PROJECT=""
WANDB_ENTITY=""
while (( $# )); do
    case "$1" in
        --existing-results)
            if (( $# < 2 )); then
                echo "missing value for --existing-results" >&2
                exit 2
            fi
            EXISTING_RESULTS="$2"
            shift 2
            ;;
        --log-to-wandb) LOG_TO_WANDB=true; shift ;;
        --wandb-project)
            if (( $# < 2 )); then
                echo "missing value for --wandb-project" >&2
                exit 2
            fi
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-entity)
            if (( $# < 2 )); then
                echo "missing value for --wandb-entity" >&2
                exit 2
            fi
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --) shift; break ;;
        *)
            echo "unknown batch option: $1" >&2
            exit 2
            ;;
    esac
done
EVAL_ARGS=("$@")

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

source "$FASTKVZIP_VENV/bin/activate"
cd "$PROJECT_DIR"

readonly DATASETS_VERSION="$(python -c 'import datasets; print(datasets.__version__)')"
if [[ "$DATASETS_VERSION" != "4.0.0" ]]; then
    echo "datasets==4.0.0 is required; found $DATASETS_VERSION" >&2
    echo "upgrade once with: uv pip install --python $FASTKVZIP_VENV/bin/python --no-deps datasets==4.0.0" >&2
    exit 2
fi

echo "commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
echo "results=$RUN_DIR"
echo "metrics=$METRICS_PATH"
python --version
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python -u prefill/eval_graph.py \
    "${EVAL_ARGS[@]}" \
    --run-dir "$RUN_DIR" \
    --existing-results "$EXISTING_RESULTS"

METRIC_ARGS=(--run-dir "$RUN_DIR")
if [[ "$LOG_TO_WANDB" == true ]]; then
    METRIC_ARGS+=(--log-to-wandb --wandb-project "$WANDB_PROJECT")
    if [[ -n "$WANDB_ENTITY" ]]; then
        METRIC_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

PYTHONPATH="$PROJECT_DIR/prefill${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m results.parse "${METRIC_ARGS[@]}"
