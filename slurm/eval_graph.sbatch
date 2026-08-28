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

readonly RUN_NAME="${1:?usage: sbatch ... eval_graph.sbatch RUN_NAME [eval_graph.py arguments]}"
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
readonly METRICS_DIR="$RESULTS_ROOT/metrics"
readonly METRICS_PATH="$METRICS_DIR/$RUN_NAME.txt"
EVAL_ARGS=("$@")

EXISTING_RESULT=""
if [[ -d "$RESULTS_ROOT" ]]; then
    EXISTING_RESULT="$(find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -type d \
        -name "*_graph_$RUN_NAME" -print -quit)"
fi
if [[ -n "$EXISTING_RESULT" || -e "$METRICS_PATH" ]]; then
    echo "evaluation run already exists: $RUN_NAME" >&2
    [[ -n "$EXISTING_RESULT" ]] && echo "result: $EXISTING_RESULT" >&2
    [[ -e "$METRICS_PATH" ]] && echo "metrics: $METRICS_PATH" >&2
    exit 2
fi

DATA_NAME="squad"
LEVEL="pair"
RATIOS=()
ARG_INDEX=0
while (( ARG_INDEX < ${#EVAL_ARGS[@]} )); do
    ARG="${EVAL_ARGS[$ARG_INDEX]}"
    case "$ARG" in
        -d|--data|--level)
            if (( ARG_INDEX + 1 >= ${#EVAL_ARGS[@]} )); then
                echo "missing value for $ARG" >&2
                exit 2
            fi
            if [[ "$ARG" == "-d" || "$ARG" == "--data" ]]; then
                DATA_NAME="${EVAL_ARGS[$((ARG_INDEX + 1))]}"
            else
                LEVEL="${EVAL_ARGS[$((ARG_INDEX + 1))]}"
            fi
            ARG_INDEX=$((ARG_INDEX + 2))
            continue
            ;;
        --data=*) DATA_NAME="${ARG#*=}" ;;
        --level=*) LEVEL="${ARG#*=}" ;;
        --ratios)
            RATIOS=()
            ARG_INDEX=$((ARG_INDEX + 1))
            while (( ARG_INDEX < ${#EVAL_ARGS[@]} )) && \
                [[ "${EVAL_ARGS[$ARG_INDEX]}" != -* ]]; do
                RATIOS+=("${EVAL_ARGS[$ARG_INDEX]}")
                ARG_INDEX=$((ARG_INDEX + 1))
            done
            continue
            ;;
        --ratios=*)
            RATIOS=("${ARG#*=}")
            ARG_INDEX=$((ARG_INDEX + 1))
            while (( ARG_INDEX < ${#EVAL_ARGS[@]} )) && \
                [[ "${EVAL_ARGS[$ARG_INDEX]}" != -* ]]; do
                RATIOS+=("${EVAL_ARGS[$ARG_INDEX]}")
                ARG_INDEX=$((ARG_INDEX + 1))
            done
            continue
            ;;
    esac
    ARG_INDEX=$((ARG_INDEX + 1))
done

source "$FASTKVZIP_VENV/bin/activate"
cd "$PROJECT_DIR"

readonly DATASETS_VERSION="$(python -c 'import datasets; print(datasets.__version__)')"
if [[ "$DATASETS_VERSION" != "4.0.0" ]]; then
    echo "datasets==4.0.0 is required; found $DATASETS_VERSION" >&2
    echo "upgrade once with: uv pip install --python $FASTKVZIP_VENV/bin/python --no-deps datasets==4.0.0" >&2
    exit 2
fi

echo "commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
echo "results_root=$RESULTS_ROOT"
echo "metrics=$METRICS_PATH"
python --version
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python -u prefill/eval_graph.py "${EVAL_ARGS[@]}" --tag "$RUN_NAME"

readonly FIRST_RESULT="$(find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -type d \
    -name "*_graph_$RUN_NAME" -print -quit)"
if [[ -z "$FIRST_RESULT" ]]; then
    echo "evaluation produced no results for run: $RUN_NAME" >&2
    exit 1
fi
readonly RESULT_FOLDER="$(basename -- "$FIRST_RESULT")"
readonly RESULT_MODEL="${RESULT_FOLDER#*_}"

METRIC_ARGS=(
    --model "$RESULT_MODEL"
    --data "$DATA_NAME"
    --level "$LEVEL"
)
if (( ${#RATIOS[@]} )); then
    METRIC_ARGS+=(--ratios "${RATIOS[@]}")
fi

mkdir -p "$METRICS_DIR"
readonly METRICS_TEMP="$METRICS_DIR/.$RUN_NAME.$SLURM_JOB_ID.tmp"
trap 'rm -f -- "$METRICS_TEMP"' EXIT
PYTHONPATH="$PROJECT_DIR/prefill${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m results.parse "${METRIC_ARGS[@]}" | tee "$METRICS_TEMP"
mv "$METRICS_TEMP" "$METRICS_PATH"
trap - EXIT
