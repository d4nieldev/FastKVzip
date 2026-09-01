#!/usr/bin/env bash
set -euo pipefail

export EVAL_GRAPH_SCRIPT=prefill/eval_graph_chunked.py
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/submit_eval_graph.sh" "$@"
