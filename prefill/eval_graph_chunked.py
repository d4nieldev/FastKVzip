"""Evaluate a graph FastKVzip checkpoint with paper-style chunk pruning."""

import sys
from collections import defaultdict
from time import perf_counter

import torch
from tqdm import tqdm

from data import DataWrapper, load_dataset_all
from eval import get_data_list, set_ratios
from eval_graph import (
    _example_output,
    _postfix,
    _prepared_full_answers,
    _record_phase_percentages,
    build_parser as _build_parser,
)
from graph import resolve_graph_microbatch_size
from graph.evaluation import (
    build_evaluation_runtime,
    load_evaluation_checkpoint,
    restore_checkpoint_prefix,
    score_context_chunk_cache,
)
from results.evaluation_run import EvaluationRun
from results.parse import finalize_task
from utils import Evaluator, set_gen_length


def build_parser():
    parser = _build_parser()
    parser.description = __doc__
    return parser


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
        prefill_mode="chunked",
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
            max_tokens = 0
            phase_percentages = [[], [], []]
            cuda.reset_peak_memory_stats(device)
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
                                        max_tokens,
                                        phase_percentages,
                                        cuda.max_memory_allocated(device),
                                        gpu_capacity,
                                    ),
                                    refresh=False,
                                )
                                task_progress.update(1)
                                continue

                            cuda.synchronize(device)
                            task_progress.set_postfix(
                                _postfix(
                                    max_tokens,
                                    phase_percentages,
                                    cuda.max_memory_allocated(device),
                                    gpu_capacity,
                                )
                            )

                            total_start = clock()
                            prefill_seconds = 0.0
                            mixer_seconds = 0.0
                            generation_seconds = 0.0
                            inputs = info = evaluator = None

                            if needs_full_answer:
                                prefill_start = clock()
                                full_kv = dataset.prefill_context(
                                    data_idx,
                                    prefill_chunk=checkpoint.prefill_chunk,
                                    save_hidden=False,
                                    do_score=False,
                                )
                                cuda.synchronize(device)
                                prefill_seconds += clock() - prefill_start
                                max_tokens = max(max_tokens, full_kv.ctx_len)
                                task_progress.set_postfix(
                                    _postfix(
                                        max_tokens,
                                        phase_percentages,
                                        cuda.max_memory_allocated(device),
                                        gpu_capacity,
                                    )
                                )

                                generation_start = clock()
                                inputs, info = dataset.generate_answer(
                                    data_idx,
                                    full_kv,
                                    prob=False,
                                    full_cache_answer=True,
                                )
                                evaluator = evaluator_factory(model, inputs, info)
                                cuda.synchronize(device)
                                generation_seconds += clock() - generation_start
                                del full_kv

                            if not ratios_to_run:
                                full_answers = _prepared_full_answers(evaluator)
                                evaluation_run.merge_example(
                                    data_name,
                                    data_idx,
                                    outputs=None,
                                    full_answers=full_answers,
                                )

                            for ratio in ratios_to_run:
                                ratio_mixer_seconds = 0.0

                                def score_chunk(kv):
                                    nonlocal ratio_mixer_seconds
                                    cuda.synchronize(device)
                                    mixer_start = clock()
                                    score_context_chunk_cache(
                                        kv,
                                        scorer,
                                        token_microbatch_size=(
                                            kv.hidden_cache[0].size(1)
                                            if token_microbatch_size == "full"
                                            else token_microbatch_size
                                            or checkpoint.token_microbatch_size
                                        ),
                                        graph_microbatch_size=graph_microbatch_size,
                                    )
                                    cuda.synchronize(device)
                                    ratio_mixer_seconds += clock() - mixer_start

                                ratio_prefill_start = clock()
                                kv = dataset.prefill_context(
                                    data_idx,
                                    prefill_chunk=checkpoint.prefill_chunk,
                                    window_size=args.window_size,
                                    chunk_ratio=ratio,
                                    level=args.level,
                                    save_hidden=True,
                                    do_score=False,
                                    chunk_scorer=score_chunk,
                                )
                                cuda.synchronize(device)
                                ratio_prefill_seconds = clock() - ratio_prefill_start
                                mixer_seconds += ratio_mixer_seconds
                                prefill_seconds += (
                                    ratio_prefill_seconds - ratio_mixer_seconds
                                )
                                max_tokens = max(max_tokens, kv.ctx_len)
                                task_progress.set_postfix(
                                    _postfix(
                                        max_tokens,
                                        phase_percentages,
                                        cuda.max_memory_allocated(device),
                                        gpu_capacity,
                                    )
                                )

                                generation_start = clock()
                                if evaluator is None:
                                    inputs, info = dataset.generate_answer(
                                        data_idx,
                                        kv,
                                        prob=False,
                                        full_cache_answer=False,
                                    )
                                    evaluator = evaluator_factory(model, inputs, info)
                                true_ratio = kv.valid.float().mean().item()
                                ratio_outputs = defaultdict(list)
                                for fmt, value in evaluator(
                                    kv, generate=True
                                ).items():
                                    ratio_outputs[fmt].append(
                                        [
                                            [
                                                ratio,
                                                round(true_ratio, 4),
                                                0.0,
                                            ],
                                            value,
                                        ]
                                    )
                                cuda.synchronize(device)
                                generation_seconds += clock() - generation_start
                                evaluation_run.merge_example(
                                    data_name,
                                    data_idx,
                                    outputs=ratio_outputs,
                                )
                                del kv

                            total_seconds = clock() - total_start
                            _record_phase_percentages(
                                phase_percentages,
                                (prefill_seconds, mixer_seconds, generation_seconds),
                                total_seconds,
                            )
                            task_peak_allocated = cuda.max_memory_allocated(device)
                            task_progress.set_postfix(
                                _postfix(
                                    max_tokens,
                                    phase_percentages,
                                    task_peak_allocated,
                                    gpu_capacity,
                                ),
                                refresh=False,
                            )
                            task_progress.update(1)
                            if verbose:
                                print(
                                    f"## Time: {total_seconds:.1f}s. Task peak GPU: "
                                    f"{task_peak_allocated / 2**30:.1f}/{gpu_capacity / 2**30:.1f}GiB. "
                                    f"[{data_name}-{data_idx}]"
                                )
                            del inputs, info, evaluator
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
