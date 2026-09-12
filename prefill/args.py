import argparse
from pathlib import Path

from window import parse_window_size
from generation import add_generation_arguments

parser = argparse.ArgumentParser(description="")
add_generation_arguments(parser)
# Method
parser.add_argument("-g", "--gate_path_or_name", type=str, default="fastkvzip")
parser.add_argument("--prefill_chunk", type=int, default=16000)
parser.add_argument(
    "--window-size",
    "--window_size",
    type=parse_window_size,
    default=4096,
    help="protected token count, or context ratio between 0 and 1",
)
parser.add_argument(
    "-r", "--ratio", type=float, default=0.3, help="compression ratio (= retained/full)"
)
parser.add_argument(
    "--kv_type",
    type=str,
    default="retain",
    choices=["evict", "retain"],
    help="retain: store full cache in storage, evict: delete the evicted KV from store.",
)
parser.add_argument(
    "--level",
    type=str,
    default="",
    choices=["pair", "pair-head", "pair-layer", "adakv-layer", ""],
    help="Eviction structure. pair-head/layer: uniform head/layer-budget; adakv-layer: with safeguard",
)
# Model and Data
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="squad",
    help="check the dataset list in data/load.py (e.g., squad, scbench_kv)",
)
parser.add_argument("--idx", type=int, default=0, help="the index of a data example")
parser.add_argument(
    "--num", type=int, default=100, help="the total number of eval data"
)
parser.add_argument("--tag", type=str, default="", help="evaluation folder name tag")
parser.add_argument("--run-dir", type=Path, help="resumable evaluation result directory")
parser.add_argument("--answer-cache-dir", type=Path, help="reuse compatible full-cache summary samples across runs")
parser.add_argument("--existing-results", choices=("fail", "resume", "overwrite"), default="fail")
parser.add_argument("--full-cache-answer", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ratios", type=float, nargs="+")
parser.add_argument("--ruler-prompt-mode", choices=("graphkv", "official"), default="graphkv")
parser.add_argument("--wandb-run-id")
parser.add_argument("--log-to-wandb", action="store_true")
parser.add_argument("--wandb-project")
parser.add_argument("--wandb-entity")


def parse_args(argv=None, *, num_default=100):
    parser.set_defaults(num=num_default)
    args = parser.parse_args(argv)

    if args.level == "":
        # Use default eviction structure setting
        if "expect" in args.gate_path_or_name:
            args.level = "adakv-layer"
        elif "snap" in args.gate_path_or_name:
            args.level = "pair-head"
        else:
            args.level = "pair"

    if args.tag:
        args.tag = f"_{args.tag}"
    if args.gate_path_or_name:
        args.tag = "_" + args.gate_path_or_name.split("/")[-1] + args.tag
    return args
