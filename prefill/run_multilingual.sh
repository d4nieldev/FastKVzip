#!/usr/bin/env bash
# Multilingual SQuAD evaluation (zh, ko, de, fr, es)
# Runs 5 methods in parallel across 8 GPUs
#
# Usage:
#   bash run_multilingual.sh Qwen/Qwen2.5-7B-Instruct-1M 0

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct-1M"}
GPU_START=${2:-0}

echo "Multilingual SQuAD Evaluation"
echo "Model: $MODEL | GPUs: $GPU_START-$((GPU_START+7))"
echo "=============================================="

LANGS=(zh ko de fr es)

# KVzip (full prefill) — 2 GPUs
run_kvzip() {
    local gpu=$1
    shift
    for LANG in "$@"; do
        CUDA_VISIBLE_DEVICES=$gpu python -B eval.py \
            -g "" -m $MODEL -d multilingual_$LANG --num 100
    done
}

# Chunked-prefill methods
run_chunk() {
    local gate=$1 gpu=$2
    shift 2
    for LANG in "$@"; do
        CUDA_VISIBLE_DEVICES=$gpu python -B eval_chunk.py \
            -g $gate -m $MODEL -d multilingual_$LANG --num 100
    done
}

# KVzip: GPU 0-1 (split languages)
run_kvzip $((GPU_START))   zh ko es   &
run_kvzip $((GPU_START+1)) de fr      &

# FastKVzip: GPU 2-3 (split languages)
run_chunk fastkvzip $((GPU_START+2)) zh ko es &
run_chunk fastkvzip $((GPU_START+3)) de fr    &

# DuoAttention: GPU 4-5 (split languages)
run_chunk head $((GPU_START+4)) zh ko es &
run_chunk head $((GPU_START+5)) de fr    &

# ExpectedAttention: GPU 6
run_chunk expect $((GPU_START+6)) zh ko de fr es &

# SnapKV: GPU 7
run_chunk snap $((GPU_START+7)) zh ko de fr es &

wait
echo ""
echo "=============================================="
echo "All done. Parse results:"
echo "  python -B -m results.parse -m {model_tag} -d multilingual"
echo "=============================================="
