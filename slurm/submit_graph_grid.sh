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
  --dry-run          print all sbatch commands without submitting
EOF
}

GPU=""
TIME=""
MEM=""
DRY_RUN=false
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
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$GPU" || -z "$TIME" || -z "$MEM" ]]; then
    usage >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
readonly BATCH_SCRIPT="$PROJECT_DIR/slurm/train_graph.sbatch"
readonly CACHE_DIR="/groups/ydar_group/danieloh/fastkvzip-implicit/teacher-cache-qwen3-8b-prefill16k"
readonly OUTPUT_ROOT="$PROJECT_DIR/graph_checkpoints"
readonly LOG_DIR="$PROJECT_DIR/.slurm/logs"
readonly MODEL_ID="Qwen/Qwen3-8B"
readonly FASTKVZIP_VENV="${FASTKVZIP_VENV:-/home/danieloh/.venvs/fastkvzip}"
readonly JOB_EXPORTS="ALL,MODEL_ID=$MODEL_ID,FASTKVZIP_VENV=$FASTKVZIP_VENV,OUTPUT_ROOT=$OUTPUT_ROOT,TEACHER_CACHE_DIR=$CACHE_DIR"

readonly -a EPOCHS=(1 1 1 2 2 2 4 4 4)
readonly -a GATES=(random pretrained pretrained random pretrained pretrained random pretrained pretrained)
readonly -a FROZEN=(false false true false false true false false true)

RUN_NAMES=()
for index in "${!EPOCHS[@]}"; do
    state=trainable
    if [[ "${FROZEN[$index]}" == true ]]; then
        state=frozen
    fi
    RUN_NAMES+=("qwen3-8b-e${EPOCHS[$index]}-gate-${GATES[$index]}-${state}-seed0")
done

existing=()
for run_name in "${RUN_NAMES[@]}"; do
    if [[ -e "$OUTPUT_ROOT/$run_name" ]]; then
        existing+=("$OUTPUT_ROOT/$run_name")
    fi
done
if (( ${#existing[@]} )); then
    echo "refusing to overwrite existing checkpoint directories:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    exit 2
fi

cache_is_complete() {
    local index
    for ((index = 0; index <= 31; index++)); do
        [[ -f "$CACHE_DIR/fineweb_10k/$index.pt" ]] || return 1
    done
    for ((index = 0; index <= 5; index++)); do
        [[ -f "$CACHE_DIR/fineweb_10k_cat/$index.pt" ]] || return 1
    done
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

LAST_JOB_ID=""
submit_config() {
    local index="$1"
    local dependency="${2:-}"
    local run_name="${RUN_NAMES[$index]}"
    local -a train_args=(
        --epochs "${EPOCHS[$index]}"
        --seed 0
        --save-strategy epochs --save-every 1
        --eval-strategy epochs --eval-every 1
        --no-save-best
        --token-microbatch-size 16000
        --graph-microbatch-size 16
    )
    if [[ "${GATES[$index]}" == random ]]; then
        train_args+=(--no-freeze-gate)
    elif [[ "${FROZEN[$index]}" == true ]]; then
        train_args+=(--gate-checkpoint fastkvzip --freeze-gate)
    else
        train_args+=(--gate-checkpoint fastkvzip --no-freeze-gate)
    fi

    local -a command=(
        sbatch --parsable
        --job-name="$run_name"
        --output="$LOG_DIR/%j-$run_name.log"
        --gpus="$GPU"
        --time="$TIME"
        --mem="$MEM"
        --export="$JOB_EXPORTS"
    )
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="afterok:$dependency")
    fi
    command+=("$BATCH_SCRIPT" "$run_name" "${train_args[@]}")

    if [[ "$DRY_RUN" == true ]]; then
        print_command "${command[@]}"
        LAST_JOB_ID="DRY_RUN_$index"
    else
        local submission
        submission="$("${command[@]}")"
        LAST_JOB_ID="${submission%%;*}"
        if ! [[ "$LAST_JOB_ID" =~ ^[0-9]+$ ]]; then
            echo "could not parse job ID: $submission" >&2
            exit 1
        fi
    fi
    printf '%s job_id=%s name=%s dependency=%s\n' \
        "$( [[ "$DRY_RUN" == true ]] && printf planned || printf submitted )" \
        "$LAST_JOB_ID" "$run_name" "${dependency:-none}"
}

cd "$PROJECT_DIR"

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$LOG_DIR"
fi

if cache_is_complete; then
    echo "teacher_cache=warm path=$CACHE_DIR"
    for index in "${!RUN_NAMES[@]}"; do
        submit_config "$index"
    done
else
    echo "teacher_cache=cold_or_partial path=$CACHE_DIR"
    submit_config 0
    readonly CACHE_JOB_ID="$LAST_JOB_ID"
    for index in "${!RUN_NAMES[@]}"; do
        (( index == 0 )) && continue
        submit_config "$index" "$CACHE_JOB_ID"
    done
fi
