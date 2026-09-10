# ==============================================================================
# Benchmark evaluation with KV eviction after prefill
# ==============================================================================
from collections import defaultdict
from contextlib import nullcontext

from data.benchmarks import dataset_revisions, get_data_list


def set_ratios():
    ratios = [0.75, 0.5, 0.4, 0.3, 0.2]
    return ratios


def run_evaluation(
    args,
    *,
    chunked=False,
    model_factory=None,
    dataset_loader=None,
    wrapper_factory=None,
    evaluator_factory=None,
    generation_length_setter=None,
    metrics_finalizer=None,
):
    from model import ModelKVzip
    from data import DataWrapper, load_dataset_all
    from results.evaluation_run import EvaluationRun
    from results.parse import finalize_task
    from utils import Evaluator, TimeStamp, save_result, set_gen_length

    ratios = list(dict.fromkeys(args.ratios or set_ratios()))
    if args.idx < 0 or (args.num is not None and args.num < 0):
        raise ValueError("evaluation idx and num must be non-negative")
    if any(not 0 < ratio < 1 for ratio in ratios):
        raise ValueError("retention ratios must be between 0 and 1")
    if args.log_to_wandb and not (
        args.run_dir and args.wandb_run_id and args.wandb_project
    ):
        raise ValueError(
            "--log-to-wandb requires --run-dir, --wandb-run-id and --wandb-project"
        )
    if not args.log_to_wandb and (args.wandb_project or args.wandb_entity):
        raise ValueError("--wandb-project and --wandb-entity require --log-to-wandb")
    if not args.run_dir and (args.wandb_run_id or args.existing_results != "fail"):
        raise ValueError("W&B binding and resume require --run-dir")

    if chunked:
        args.tag += f"_chunk{args.prefill_chunk//1000}k_w{args.window_size}"
    elif args.gate_path_or_name:
        args.tag += f"_w{args.window_size}"
    print(f"tag: {args.tag}")

    if not chunked:
        args.kv_type = "retain"  # Evaluate all ratios from one full prefill.
    model = (model_factory or ModelKVzip)(
        args.model, args.kv_type, args.gate_path_or_name
    )
    data_names = get_data_list(args.data)
    run = nullcontext(None)
    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        run = EvaluationRun.open(
            run_dir.parent,
            run_dir.name,
            checkpoint_path=None,
            model_identity={
                "model_id": args.model,
                "model_revision": (
                    getattr(model.model, "_fastkvzip_revision", None) or "unknown"
                ),
                "gate": args.gate_path_or_name,
                "prefill_chunk": str(args.prefill_chunk),
                "kv_type": args.kv_type,
            },
            wandb_run_id=args.wandb_run_id,
            window_size=args.window_size,
            level=args.level,
            prefill_mode="chunked" if chunked else "post-prefill",
            ruler_prompt_mode=args.ruler_prompt_mode,
            dataset_revisions=dataset_revisions(data_names),
            existing_results=args.existing_results,
        )

    with run as evaluation_run:
        for args.data in data_names:
            rows = (dataset_loader or load_dataset_all)(
                args.data, model.tokenizer,
                n_data=None if args.num is None else args.idx + args.num,
            )
            dataset = (wrapper_factory or DataWrapper)(
                args.data, rows, model, ruler_prompt_mode=args.ruler_prompt_mode,
            )
            (generation_length_setter or set_gen_length)(args.data, model)
            task = args.data
            if task.startswith("ruler_") and args.ruler_prompt_mode == "official":
                task += "_official"
            dataset_size = getattr(rows, "full_size", None)
            if evaluation_run:
                evaluation_run.record_dataset_size(task, dataset_size)

            tt = TimeStamp(True)
            max_idx = (
                len(dataset) if args.num is None
                else min(args.idx + args.num, len(dataset))
            )
            print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

            for data_idx in range(args.idx, max_idx):
                existing = (
                    evaluation_run.load_example(task, data_idx)
                    if evaluation_run else None
                )
                remaining = [
                    r for r in ratios
                    if existing is None or r not in existing.requested_ratios
                ]
                needs_full = args.full_cache_answer and (
                    existing is None or not existing.has_full_answers
                )
                if not remaining and not needs_full:
                    continue
                kv = None
                if not chunked:
                    kv = dataset.prefill_context(
                        data_idx, window_size=args.window_size,
                        do_score=bool(remaining),
                    )
                elif needs_full:
                    kv = dataset.prefill_context(data_idx, do_score=False)
                inputs, info = dataset.generate_answer(
                    data_idx, kv, prob=False, full_cache_answer=needs_full
                )
                evaluator = (evaluator_factory or Evaluator)(model, inputs, info)
                if chunked:
                    del kv
                if not remaining:
                    evaluation_run.merge_example(task, data_idx, full_answers={
                        fmt: evaluator.decode(inputs[fmt]["a"]) for fmt in info
                    })

                outputs = defaultdict(list)
                for ratio in remaining:
                    if chunked:
                        kv = dataset.prefill_context(
                            data_idx, prefill_chunk=args.prefill_chunk,
                            window_size=args.window_size, chunk_ratio=ratio, level=args.level,
                        )
                        thres = 0
                        if args.kv_type == "evict":
                            kept = sum(
                                lengths.sum().item() for lengths in kv.info["len_k"]
                            )
                            pairs = kv.n_layers * kv.n_heads_kv
                            ratio_true = (kept - kv.sink * pairs) / (kv.ctx_len * pairs)
                        else:
                            ratio_true = kv.valid.float().mean().item()
                    else:
                        thres, ratio_true = kv.prune(ratio, args.level)
                    ratio_outputs = {}
                    for fmt, value in evaluator(kv, generate=True).items():
                        ratio_outputs[fmt] = [[
                            [ratio, round(ratio_true, 4), round(thres, 4)], value
                        ]]
                        outputs[fmt].extend(ratio_outputs[fmt])
                    if evaluation_run:
                        evaluation_run.merge_example(task, data_idx, outputs=ratio_outputs)
                    if chunked:
                        del kv

                if not evaluation_run:
                    save_result(model.name, args, outputs, data_idx)
                if not chunked:
                    del kv
                tt(f"[{args.data}-{data_idx}]\n")
                del inputs, info, evaluator

            if evaluation_run:
                (metrics_finalizer or finalize_task)(
                    evaluation_run, task, dataset_size,
                    log_to_wandb=args.log_to_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                )
            print("Finished.")


def main(argv=None, *, chunked=False):
    from args import parse_args

    run_evaluation(parse_args(argv, num_default=None), chunked=chunked)


if __name__ == "__main__":
    main()
