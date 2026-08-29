"""Evaluate a whole-context graph FastKVzip checkpoint."""

import argparse
import io
import sys
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from data import DataWrapper, load_dataset_all
from eval import get_data_list, set_ratios
from graph import resolve_graph_microbatch_size
from graph.evaluation import (
    build_evaluation_runtime,
    load_evaluation_checkpoint,
    restore_checkpoint_prefix,
    score_context_cache,
)
from results.evaluation_run import EvaluationRun
from results.parse import finalize_task
from utils import Evaluator, set_gen_length


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show detailed per-example evaluation output",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="store a resumable evaluation run in this directory",
    )
    parser.add_argument(
        "--existing-results",
        choices=("fail", "resume", "overwrite"),
        default="fail",
        help="how to handle an existing --run-dir",
    )
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
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


def _prepared_full_answers(evaluator) -> dict[str, str]:
    answers = {}
    for fmt in evaluator.info:
        full_ids = evaluator.inputs[fmt]["a"]
        if full_ids is None:
            continue
        answers[fmt] = evaluator.decode(full_ids)
    return answers


@contextmanager
def _example_output(verbose: bool):
    if verbose:
        yield None
        return
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        yield captured


def _postfix(tokens, prefill, mixer, generation, total, peak, capacity):
    return {
        "tokens": tokens,
        "prefill": prefill,
        "mixer": mixer,
        "gen": generation,
        "total": total,
        "gpu": f"{peak / 2**30:.1f}/{capacity / 2**30:.1f}GiB",
    }


def run_evaluation(
    args,
    *,
    model_factory=None,
    dataset_loader=None,
    wrapper_factory=None,
    evaluator_factory=None,
    generation_length_setter=None,
    progress_factory=tqdm,
    clock=perf_counter,
    cuda=torch.cuda,
    metrics_finalizer=finalize_task,
) -> None:
    ratios = list(dict.fromkeys(getattr(args, "ratios", None) or set_ratios()))
    log_to_wandb = getattr(args, "log_to_wandb", False)
    wandb_project = getattr(args, "wandb_project", None)
    wandb_entity = getattr(args, "wandb_entity", None)
    if args.idx < 0 or args.num < 0:
        raise ValueError("evaluation idx and num must be non-negative")
    if log_to_wandb and not wandb_project:
        raise ValueError("--log-to-wandb requires --wandb-project")
    if not log_to_wandb and (wandb_project or wandb_entity):
        raise ValueError("--wandb-project and --wandb-entity require --log-to-wandb")

    checkpoint = load_evaluation_checkpoint(
        args.graph_checkpoint, model_override=getattr(args, "model", None)
    )
    wandb_run_id = checkpoint.payload.get("wandb_run_id")
    if log_to_wandb and (not isinstance(wandb_run_id, str) or not wandb_run_id):
        raise ValueError("--log-to-wandb requires a checkpoint with a W&B run ID")
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
    dataset_loader = dataset_loader or load_dataset_all
    wrapper_factory = wrapper_factory or DataWrapper
    evaluator_factory = evaluator_factory or Evaluator
    generation_length_setter = generation_length_setter or set_gen_length
    verbose = getattr(args, "verbose", False)

    run_dir = args.run_dir.expanduser().resolve()
    with EvaluationRun.open(
        run_dir.parent,
        run_dir.name,
        checkpoint_path=args.graph_checkpoint,
        wandb_run_id=wandb_run_id,
        window_size=args.window_size,
        level=args.level,
        existing_results=args.existing_results,
    ) as evaluation_run:
        model, scorer = build_evaluation_runtime(
            checkpoint, model_factory=model_factory
        )
        data_names = get_data_list(args.data, model.name)
        device = scorer.device
        gpu_capacity = cuda.get_device_properties(device).total_memory
        for task_index, data_name in enumerate(data_names, start=1):
            args.data = data_name
            dataset = wrapper_factory(
                data_name, dataset_loader(data_name, model.tokenizer), model
            )
            restore_checkpoint_prefix(model, checkpoint.prefix_ids)
            generation_length_setter(data_name, model)

            max_idx = min(args.idx + args.num, len(dataset))
            if verbose:
                print(
                    "=" * 80,
                    f"\nStart evaluation with {args.idx}~{max_idx} samples",
                )
            task_progress = progress_factory(
                total=max(0, max_idx - args.idx),
                desc=f"[{task_index}/{len(data_names)}] {data_name}",
            )
            try:
                for data_idx in range(args.idx, max_idx):
                    captured = None
                    try:
                        with _example_output(verbose) as captured:
                            ratios_to_run = list(ratios)
                            needs_full_answer = args.full_cache_answer
                            existing = evaluation_run.load_example(
                                data_name,
                                data_idx,
                            )
                            if existing is not None:
                                ratios_to_run = [
                                    ratio
                                    for ratio in ratios
                                    if ratio not in existing.requested_ratios
                                ]
                                needs_full_answer = (
                                    args.full_cache_answer
                                    and not existing.has_full_answers
                                )

                            if not ratios_to_run and not needs_full_answer:
                                task_progress.set_postfix(
                                    _postfix(
                                        "cached",
                                        "0.0s",
                                        "0.0s",
                                        "0.0s",
                                        "0.0s",
                                        0,
                                        gpu_capacity,
                                    ),
                                    refresh=False,
                                )
                                task_progress.update(1)
                                continue

                            cuda.synchronize(device)
                            cuda.reset_peak_memory_stats(device)
                            task_progress.set_postfix(
                                _postfix(
                                    "...",
                                    "...",
                                    "--",
                                    "--",
                                    "...",
                                    cuda.max_memory_allocated(device),
                                    gpu_capacity,
                                )
                            )

                            total_start = prefill_start = clock()
                            kv = dataset.prefill_context(
                                data_idx,
                                prefill_chunk=checkpoint.prefill_chunk,
                                save_hidden=bool(ratios_to_run),
                                do_score=False,
                            )
                            cuda.synchronize(device)
                            prefill_seconds = clock() - prefill_start
                            task_progress.set_postfix(
                                _postfix(
                                    kv.ctx_len,
                                    f"{prefill_seconds:.1f}s",
                                    "..." if ratios_to_run else "--",
                                    "--" if ratios_to_run else "...",
                                    "...",
                                    cuda.max_memory_allocated(device),
                                    gpu_capacity,
                                )
                            )

                            mixer_seconds = 0.0
                            if ratios_to_run:
                                mixer_start = clock()
                                score_context_cache(
                                    kv,
                                    scorer,
                                    prefill_chunk=checkpoint.prefill_chunk,
                                    window_size=args.window_size,
                                    token_microbatch_size=(
                                        kv.end_idx - kv.start_idx
                                        if token_microbatch_size == "full"
                                        else token_microbatch_size
                                        or checkpoint.token_microbatch_size
                                    ),
                                    graph_microbatch_size=graph_microbatch_size,
                                )
                                cuda.synchronize(device)
                                mixer_seconds = clock() - mixer_start
                                task_progress.set_postfix(
                                    _postfix(
                                        kv.ctx_len,
                                        f"{prefill_seconds:.1f}s",
                                        f"{mixer_seconds:.1f}s",
                                        "...",
                                        "...",
                                        cuda.max_memory_allocated(device),
                                        gpu_capacity,
                                    )
                                )

                            generation_start = clock()
                            inputs, info = dataset.generate_answer(
                                data_idx,
                                kv,
                                prob=False,
                                full_cache_answer=needs_full_answer,
                            )
                            evaluator = evaluator_factory(model, inputs, info)

                            if not ratios_to_run:
                                full_answers = _prepared_full_answers(evaluator)
                                evaluation_run.merge_example(
                                    data_name,
                                    data_idx,
                                    outputs=None,
                                    full_answers=full_answers,
                                )

                            for ratio in ratios_to_run:
                                threshold, true_ratio = kv.prune(ratio, args.level)
                                ratio_outputs = defaultdict(list)
                                for fmt, value in evaluator(
                                    kv, generate=True
                                ).items():
                                    ratio_outputs[fmt].append(
                                        [
                                            [
                                                ratio,
                                                round(true_ratio, 4),
                                                round(threshold, 4),
                                            ],
                                            value,
                                        ]
                                    )
                                evaluation_run.merge_example(
                                    data_name,
                                    data_idx,
                                    outputs=ratio_outputs,
                                )

                            cuda.synchronize(device)
                            generation_seconds = clock() - generation_start
                            total_seconds = clock() - total_start
                            peak_allocated = cuda.max_memory_allocated(device)
                            task_progress.set_postfix(
                                _postfix(
                                    kv.ctx_len,
                                    f"{prefill_seconds:.1f}s",
                                    f"{mixer_seconds:.1f}s",
                                    f"{generation_seconds:.1f}s",
                                    f"{total_seconds:.1f}s",
                                    peak_allocated,
                                    gpu_capacity,
                                ),
                                refresh=False,
                            )
                            task_progress.update(1)
                            if verbose:
                                print(
                                    f"## Time: {total_seconds:.1f}s. Peak GPU: "
                                    f"{peak_allocated / 2**30:.1f}/{gpu_capacity / 2**30:.1f}GiB. "
                                    f"[{data_name}-{data_idx}]"
                                )
                            del kv, inputs, info, evaluator
                    except BaseException:
                        task_progress.close()
                        if captured is not None:
                            sys.stderr.write(captured.getvalue())
                            sys.stderr.flush()
                        raise
            finally:
                task_progress.close()
            metrics_finalizer(
                evaluation_run,
                data_name,
                len(dataset),
                log_to_wandb=log_to_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            )
            if verbose:
                print("Finished.")


def main(argv=None) -> None:
    run_evaluation(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
