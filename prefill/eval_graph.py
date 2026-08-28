"""Evaluate a whole-context graph FastKVzip checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

from data import DataWrapper, load_dataset_all
from eval import get_data_list, set_ratios
from graph import resolve_graph_microbatch_size
from graph import evaluation as _evaluation
from graph.evaluation import (
    EvaluationCheckpoint,
    build_evaluation_runtime,
    load_evaluation_checkpoint,
    protect_local_window,
    reconstruct_graph_scorer,
    restore_checkpoint_prefix,
    score_context_cache,
    score_hidden_cache,
)
from utils import Evaluator, TimeStamp, save_result, set_gen_length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-d", "--data", default="squad")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument(
        "--window-size", "--window_size", dest="window_size", type=int, default=4096
    )
    parser.add_argument(
        "--level",
        choices=("pair", "pair-head", "pair-layer", "adakv-layer"),
        default="pair",
    )
    parser.add_argument(
        "--full-cache-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate the full-cache reference answer",
    )
    parser.add_argument("--ratios", nargs="+", type=_retention_ratio)
    parser.add_argument(
        "--token-microbatch-size",
        type=_token_microbatch_size,
        help="override the checkpoint value; use 'full' for one context-sized chunk",
    )
    parser.add_argument(
        "--graph-microbatch-size",
        type=_graph_microbatch_size,
        help="override the checkpoint value; use 'all' for every layer/head graph",
    )
    parser.add_argument("--tag", default="_graph")
    return parser


def _retention_ratio(value: str) -> float:
    ratio = float(value)
    if not 0 < ratio < 1:
        raise argparse.ArgumentTypeError("retention ratios must be between 0 and 1")
    return ratio


def _microbatch_size(value: str, maximum: str) -> int | str:
    if value == maximum:
        return value
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"microbatch size must be a positive integer or {maximum}"
        ) from error
    if size < 1:
        raise argparse.ArgumentTypeError("microbatch size must be positive")
    return size


def _token_microbatch_size(value: str) -> int | str:
    return _microbatch_size(value, "full")


def _graph_microbatch_size(value: str) -> int | str:
    return _microbatch_size(value, "all")


def _normalize_graph_tag(tag: str) -> str:
    if tag == "_graph" or tag.startswith("_graph_"):
        return tag
    suffix = tag.lstrip("_")
    return "_graph" if not suffix else f"_graph_{suffix}"


def run_evaluation(
    args,
    *,
    model_factory=None,
    dataset_loader=None,
    wrapper_factory=None,
    evaluator_factory=None,
    result_saver=None,
    generation_length_setter=None,
    timestamp_factory=None,
) -> None:
    checkpoint = load_evaluation_checkpoint(
        args.graph_checkpoint, model_override=getattr(args, "model", None)
    )
    graph_microbatch_size = getattr(args, "graph_microbatch_size", None)
    if graph_microbatch_size == "all":
        graph_microbatch_size = (
            int(checkpoint.config["num_layers"])
            * int(checkpoint.config["num_kv_heads"])
        )
    elif graph_microbatch_size is None:
        graph_microbatch_size = checkpoint.graph_microbatch_size
    resolve_graph_microbatch_size(
        graph_microbatch_size,
        int(checkpoint.config["num_layers"]),
        int(checkpoint.config["num_kv_heads"]),
    )
    token_microbatch_size = getattr(args, "token_microbatch_size", None)
    model, scorer = build_evaluation_runtime(checkpoint, model_factory=model_factory)
    dataset_loader = dataset_loader or load_dataset_all
    wrapper_factory = wrapper_factory or DataWrapper
    evaluator_factory = evaluator_factory or Evaluator
    result_saver = result_saver or save_result
    generation_length_setter = generation_length_setter or set_gen_length
    timestamp_factory = timestamp_factory or TimeStamp
    ratios = getattr(args, "ratios", None) or set_ratios()
    if args.idx < 0 or args.num < 0:
        raise ValueError("evaluation idx and num must be non-negative")
    args.tag = _normalize_graph_tag(args.tag)

    for data_name in get_data_list(args.data, model.name):
        args.data = data_name
        dataset = wrapper_factory(
            data_name, dataset_loader(data_name, model.tokenizer), model
        )
        restore_checkpoint_prefix(model, checkpoint.prefix_ids)
        generation_length_setter(data_name, model)

        timestamp = timestamp_factory(True)
        max_idx = min(args.idx + args.num, len(dataset))
        print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")
        for data_idx in range(args.idx, max_idx):
            kv = dataset.prefill_context(
                data_idx,
                prefill_chunk=checkpoint.prefill_chunk,
                save_hidden=True,
                do_score=False,
            )
            score_context_cache(
                kv,
                scorer,
                prefill_chunk=checkpoint.prefill_chunk,
                window_size=args.window_size,
                token_microbatch_size=(
                    kv.end_idx - kv.start_idx
                    if token_microbatch_size == "full"
                    else token_microbatch_size or checkpoint.token_microbatch_size
                ),
                graph_microbatch_size=graph_microbatch_size,
            )
            inputs, info = dataset.generate_answer(
                data_idx,
                kv,
                prob=False,
                full_cache_answer=args.full_cache_answer,
            )
            evaluator = evaluator_factory(model, inputs, info)

            outputs = defaultdict(list)
            for ratio in ratios:
                threshold, true_ratio = kv.prune(ratio, args.level)
                for fmt, value in evaluator(kv, generate=True).items():
                    outputs[fmt].append(
                        [
                            [ratio, round(true_ratio, 4), round(threshold, 4)],
                            value,
                        ]
                    )
            result_saver(model.name, args, outputs, data_idx)
            timestamp(f"[{data_name}-{data_idx}]\n")
            del kv, inputs, info, evaluator
        print("Finished.")


def main(argv=None) -> None:
    run_evaluation(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
