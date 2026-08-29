import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def _evaluate_answer(*args, **kwargs):
    from results.metric import evaluate_answer

    return evaluate_answer(*args, **kwargs)


def get_data_list(dataname, modelname=""):
    qa = [
        "squad",  # 203 (502)
        "gsm",  # 86 (120)
        "scbench_choice_eng",  # 119299
        "scbench_qa_eng",  # 122101
    ]
    retv = [
        "scbench_kv",  # 169428
        "scbench_prefix_suffix",  # 112635
        "scbench_repoqa",  # 72499
    ]
    redun = [
        "scbench_summary",  # 117806
        "scbench_vt",  # 124551
        "scbench_mf",  # 149860
        "scbench_many_shot",  # 26474
    ]

    if dataname == "qa":
        data_list = qa
    elif dataname == "retv":
        data_list = retv
    elif dataname == "redun":
        data_list = redun
    elif dataname == "all":
        data_list = qa + retv + redun
    else:
        data_list = [dataname]

    if any(k in modelname.lower() for k in ("qwen3", "gemma3", "gemma-3")):
        # Evaluate performance on shorter version for models that achieve near zero performance on specific tasks.
        data_list = [
            f"{x}_short" if x == "scbench_prefix_suffix" else x for x in data_list
        ]
        if not "instruct" in modelname.lower():
            data_list = [f"{x}_short" if x == "scbench_kv" else x for x in data_list]
            data_list = [f"{x}_mid" if x == "scbench_mf" else x for x in data_list]

    print(data_list)
    return data_list


def parse_answer(name):
    answers = []
    subtasks = []
    if "many_shot" in name:
        from datasets import load_dataset

        answers = []
        samples = load_dataset(
            "Jang-Hyun/SCBench-preprocessed",
            data_files=f"{name}.parquet",
            split="train",
        )
        for data in samples:
            d = []
            for q, gt in zip(data["prompts"][1:], data["ground_truth"]):
                # parse options, e.g., "(A) xxx" from gt = A
                cand = [sol for sol in q.split("\n") if f"({gt})" in sol]
                if len(cand) != 1:
                    print(f"Error: {q} {gt}")
                d.append(cand[0].strip())

            answers.append(d)

    elif "repoqa" in name:
        from datasets import load_dataset

        answers = []
        samples = load_dataset(
            "Jang-Hyun/SCBench-preprocessed",
            data_files=f"{name}.parquet",
            split="train",
        )
        for data in samples:
            d = defaultdict(list)
            d["lang"] = data["lang"]
            d["repo"] = data["repo"]
            d["func_name"] = data["func_name"]
            d["ground_truth"] = data["ground_truth"]
            answers.append(d)

            if "task" in data:
                subtasks.append(data["task"])

    elif "summary_with_needles" in name:
        from datasets import load_dataset

        answers = []
        subtasks = []
        samples = load_dataset(
            "Jang-Hyun/SCBench-preprocessed",
            data_files=f"{name}.parquet",
            split="train",
        )
        for data in samples:
            d = defaultdict(list)
            subtasks.append(data["task"])
            answers.append(data["ground_truth"])

    return answers, subtasks


def mean(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def avg_list_of_list(l):
    score = mean([mean(vals) for vals in l])
    return score


def max_list_of_list(l):
    m = max([max(vals) for vals in l])
    count = mean([mean([v >= m for v in vals]) for vals in l])
    return (m, round(count, 3))


def sum_list_of_list(l):
    score = sum([sum(vals) for vals in l])
    count = sum([len(vals) for vals in l])
    score /= count
    print(count)
    return score


def set_ratios():
    ratios = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2]
    return ratios


def get_eviction_level(name):
    if "expect" in name:
        level = "adakv-layer"
    elif "snap" in name:
        level = "pair-head"
    else:
        level = "pair"
    return level


def _ratio_key(ratio):
    return str(float(ratio))


def _load_run_outputs(run):
    tasks = defaultdict(list)
    for result in run.iter_examples():
        formats = result.formats
        ratios = {}
        for position, requested in enumerate(result.requested_ratios):
            entries = [result.payload[fmt][position] for fmt in formats]
            ratios[requested] = {
                "predictions": [entry[1]["pruned"] for entry in entries],
                "actual_retention": float(entries[0][0][1]),
            }
        tasks[result.task].append(
            {
                "index": result.example_index,
                "answers": [result.answers[fmt] for fmt in formats],
                "full_predictions": (
                    [result.full_answers[fmt] for fmt in formats]
                    if result.has_full_answers
                    else None
                ),
                "ratios": ratios,
            }
        )
    if not tasks:
        raise ValueError(f"evaluation run has no result files: {run.run_dir}")
    return dict(tasks)


def _average_relative_performance(task_metrics):
    relative_by_ratio = defaultdict(list)
    for values in task_metrics.values():
        for ratio, ratio_values in values["ratios"].items():
            if "relative" in ratio_values:
                relative_by_ratio[float(ratio)].append(ratio_values["relative"])
    return {
        _ratio_key(ratio): mean(values)
        for ratio, values in sorted(relative_by_ratio.items())
    }


def build_run_metrics(
    run,
    *,
    dataset_sizes,
    evaluate=None,
    supplementary_loader=parse_answer,
):
    evaluate = evaluate or _evaluate_answer
    task_metrics = {}

    for task_name, examples in _load_run_outputs(run).items():
        try:
            dataset_size = int(dataset_sizes[task_name])
        except KeyError as error:
            raise ValueError(f"dataset size is required for {task_name}") from error
        example_indices = {example["index"] for example in examples}
        task_complete = example_indices == set(range(dataset_size))
        supplemental_answers, subtasks = supplementary_loader(task_name)
        scores = defaultdict(list)
        actual_retention = defaultdict(list)
        full_scores = []

        for example in examples:
            index = example["index"]
            answers = example["answers"]
            if supplemental_answers:
                if index >= len(supplemental_answers):
                    raise ValueError(
                        f"missing supplemental answer for {task_name} example {index}"
                    )
                answers = supplemental_answers[index]
            subtask = None
            if subtasks:
                if index >= len(subtasks):
                    raise ValueError(
                        f"missing subtask for {task_name} example {index}"
                    )
                subtask = subtasks[index]

            for ratio, result in example["ratios"].items():
                scores[ratio].append(
                    evaluate(
                        result["predictions"],
                        answers,
                        task_name,
                        "qa",
                        subtask=subtask,
                    )
                )
                actual_retention[ratio].append(result["actual_retention"])
            if example["full_predictions"] is not None:
                full_scores.append(
                    evaluate(
                        example["full_predictions"],
                        answers,
                        task_name,
                        "qa",
                        subtask=subtask,
                    )
                )

        full_complete = task_complete and len(full_scores) == dataset_size
        full_score = avg_list_of_list(full_scores) * 100 if full_scores else None
        task_result = {
            "example_count": len(examples),
            "dataset_size": dataset_size,
            "complete": task_complete,
            "full_cache": {
                "score": full_score,
                "example_count": len(full_scores),
                "complete": full_complete,
            },
            "ratios": {},
        }
        for ratio in sorted(scores):
            score = avg_list_of_list(scores[ratio]) * 100
            ratio_result = {
                "score": score,
                "actual_retention": mean(actual_retention[ratio]),
                "example_count": len(scores[ratio]),
                "dataset_size": dataset_size,
                "complete": task_complete and len(scores[ratio]) == dataset_size,
            }
            if full_complete and full_score:
                ratio_result["relative"] = score / full_score * 100
            task_result["ratios"][_ratio_key(ratio)] = ratio_result

        if full_complete and full_score:
            task_result["ratios"]["1.0"] = {
                "score": full_score,
                "relative": 100.0,
                "actual_retention": 1.0,
                "example_count": dataset_size,
                "dataset_size": dataset_size,
                "complete": True,
            }
        task_metrics[task_name] = task_result

    return {
        "tasks": task_metrics,
        "average_relative_performance": _average_relative_performance(task_metrics),
    }


def _wandb_points(metrics):
    points = {}
    for task, task_metrics in metrics["tasks"].items():
        for ratio_key, ratio_metrics in task_metrics["ratios"].items():
            ratio = float(ratio_key)
            values = {
                f"test/{task}": ratio_metrics["score"],
                f"test/{task}-actual-retention": ratio_metrics["actual_retention"],
            }
            if "relative" in ratio_metrics:
                values[f"test/{task}-relative"] = ratio_metrics["relative"]
            for key, value in values.items():
                points[(key, _ratio_key(ratio))] = float(value)
    return points


def _require_full_benchmarks(metrics):
    for task, task_metrics in metrics["tasks"].items():
        if not task_metrics["complete"]:
            raise ValueError(f"W&B logging requires the full {task} benchmark")
        requested = [
            ratio_metrics
            for ratio, ratio_metrics in task_metrics["ratios"].items()
            if float(ratio) < 1
        ]
        if not requested:
            raise ValueError(f"W&B logging requires retention results for {task}")
        if any(not ratio_metrics["complete"] for ratio_metrics in requested):
            raise ValueError(f"W&B logging requires complete ratio coverage for {task}")


def upload_run_metrics(
    metrics,
    manifest,
    *,
    project,
    entity=None,
    wandb_module=None,
):
    _require_full_benchmarks(metrics)
    if wandb_module is None:
        import wandb as wandb_module

    run_id = manifest.get("wandb_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("evaluation run does not contain a W&B run ID")
    api = wandb_module.Api()
    resolved_entity = entity or getattr(api, "default_entity", None)
    if not resolved_entity:
        raise ValueError("W&B entity was not supplied and no default entity is configured")
    run_path = f"{resolved_entity}/{project}/{run_id}"
    remote_run = api.run(run_path)
    if str(remote_run.state).lower() != "finished":
        raise ValueError(f"W&B training run is not finished: {remote_run.state}")

    local = _wandb_points(metrics)
    remote = {}
    axis = "test/retention_ratio"
    metric_keys = {
        key
        for task in metrics["tasks"]
        for key in (
            f"test/{task}",
            f"test/{task}-relative",
            f"test/{task}-actual-retention",
        )
    }
    for metric_key in sorted(metric_keys):
        for row in remote_run.scan_history(
            keys=[axis, metric_key], page_size=10000
        ):
            if axis not in row or metric_key not in row:
                continue
            point = (metric_key, _ratio_key(row[axis]))
            if point in remote:
                raise ValueError(
                    f"duplicate W&B metric point: {metric_key} at {point[1]}"
                )
            remote[point] = float(row[metric_key])

    missing = {}
    for point, value in local.items():
        if point in remote:
            if not math.isclose(value, remote[point], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"W&B metric conflicts with local results: {point[0]} at {point[1]}"
                )
            continue
        missing[point] = value
    if not missing:
        return 0

    with tempfile.TemporaryDirectory(prefix="fastkvzip-eval-wandb-") as root_dir:
        settings = wandb_module.Settings(
            root_dir=root_dir,
            x_disable_stats=True,
            x_disable_meta=True,
        )
        run = wandb_module.init(
            project=project,
            entity=resolved_entity,
            id=run_id,
            resume="must",
            settings=settings,
        )
        try:
            run.define_metric(axis)
            for metric_key in sorted({key for key, _ in local}):
                run.define_metric(metric_key, step_metric=axis)
            by_ratio = defaultdict(dict)
            for (key, ratio), value in missing.items():
                by_ratio[float(ratio)][key] = value
            for ratio in sorted(by_ratio):
                run.log({axis: ratio, **by_ratio[ratio]})
        finally:
            run.finish(exit_code=0)
    return len(missing)


def _print_run_metrics(run_dir, metrics, level):
    for task, task_metrics in metrics["tasks"].items():
        print(
            f"\nEvaluate {task} on {task_metrics['example_count']}/"
            f"{task_metrics['dataset_size']} samples, {Path(run_dir).name}"
        )
        print(f"level: {level}")
        print("performance per requested ratio")
        for ratio, values in sorted(
            task_metrics["ratios"].items(), key=lambda item: float(item[0]), reverse=True
        ):
            relative = (
                f", relative={values['relative']:.2f}"
                if "relative" in values
                else ""
            )
            print(
                f"{ratio}: score={values['score']:.2f}, "
                f"actual={values['actual_retention']:.4f}{relative}"
            )
    print("=" * 50)
    print("Averaged relative performance (note, MRCR is not included)")
    for ratio, value in sorted(
        metrics["average_relative_performance"].items(),
        key=lambda item: float(item[0]),
        reverse=True,
    ):
        print(f"{ratio}: {value:.2f}")


def _read_metrics(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metrics from {path}") from error
    if not isinstance(metrics, dict) or not isinstance(metrics.get("tasks"), dict):
        raise ValueError(f"invalid metrics file: {path}")
    return metrics


def _task_is_complete(task_metrics):
    requested = [
        values
        for ratio, values in task_metrics["ratios"].items()
        if float(ratio) < 1
    ]
    return (
        task_metrics["complete"]
        and bool(requested)
        and all(values["complete"] for values in requested)
    )


def _load_dataset_size(task):
    from data.load import load_scbench, load_squad

    if task == "squad":
        return len(load_squad(100))
    if task == "gsm":
        return 100
    if task.startswith("scbench_"):
        return len(load_scbench(task))
    raise ValueError(f"cannot load dataset size for {task}")


def _dataset_sizes(run, previous, known=None):
    tasks = {
        path.name for path in run.outputs_dir.iterdir() if path.is_dir()
    }
    sizes = {
        task: values["dataset_size"]
        for task, values in previous.get("tasks", {}).items()
        if task in tasks
    }
    sizes.update({task: size for task, size in (known or {}).items() if task in tasks})
    for task in tasks - sizes.keys():
        sizes[task] = _load_dataset_size(task)
    return sizes


def finalize_task(
    run,
    task,
    dataset_size,
    *,
    log_to_wandb=False,
    wandb_project=None,
    wandb_entity=None,
):
    """Update metrics and W&B as soon as one concrete benchmark finishes."""
    previous = (
        _read_metrics(run.metrics_path)
        if run.metrics_path.exists()
        else {"tasks": {}, "average_relative_performance": {}}
    )
    metrics = build_run_metrics(
        run,
        dataset_sizes=_dataset_sizes(run, previous, {task: dataset_size}),
    )
    run.write_metrics(metrics)
    current = {
        "tasks": {task: metrics["tasks"][task]},
        "average_relative_performance": _average_relative_performance(
            {task: metrics["tasks"][task]}
        ),
    }
    _print_run_metrics(run.run_dir, current, run.manifest["level"])
    if log_to_wandb:
        uploaded = upload_run_metrics(
            current,
            run.manifest,
            project=wandb_project,
            entity=wandb_entity,
        )
        print(f"W&B metric points uploaded: {uploaded}")
    return metrics


def _run_directory(args):
    run_dir = args.run_dir.resolve()
    from results.evaluation_run import EvaluationRun

    with EvaluationRun.load(run_dir) as run:
        previous = (
            _read_metrics(run.metrics_path)
            if run.metrics_path.exists()
            else {"tasks": {}}
        )
        metrics = build_run_metrics(
            run,
            dataset_sizes=_dataset_sizes(run, previous),
        )
        run.write_metrics(metrics)
        _print_run_metrics(run_dir, metrics, run.manifest["level"])
        if args.log_to_wandb:
            complete = {
                "tasks": {
                    task: values
                    for task, values in metrics["tasks"].items()
                    if _task_is_complete(values)
                }
            }
            complete["average_relative_performance"] = (
                _average_relative_performance(complete["tasks"])
            )
            if not complete["tasks"]:
                raise ValueError("no complete benchmark metrics are available for W&B")
            uploaded = upload_run_metrics(
                complete,
                run.manifest,
                project=args.wandb_project,
                entity=args.wandb_entity,
            )
            print(f"W&B metric points uploaded: {uploaded}")


def retention_ratio(value):
    ratio = float(value)
    if not 0 < ratio < 1:
        raise argparse.ArgumentTypeError("retention ratios must be between 0 and 1")
    return ratio


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen2.5-7b-instruct-1m_fastkvzip_chunk16k_w4096",
    )
    parser.add_argument("-d", "--data", type=str, default="all")
    parser.add_argument("-l", "--level", type=str, default="")
    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--ratios", nargs="+", type=retention_ratio)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    return parser


def _run_legacy(args):
    import glob

    if args.level == "":
        args.level = get_eviction_level(args.model)

    ratios = [1.0, *(args.ratios or set_ratios()[1:])]
    folder_tag = f"_{args.tag}" if args.tag else ""
    args.model += folder_tag
    cur_path = "./results"

    scores_ratio_all = {r: [] for r in ratios}

    data_list = get_data_list(args.data, args.model)
    for args.data in data_list:
        answers_supp, subtasks = parse_answer(args.data)

        folder_list = glob.glob(
            os.path.join(
                cur_path, f"{args.data}/*_{args.model}/output-{args.level}.json"
            )
        )
        max_idx = len(folder_list)
        folder_list = [
            os.path.join(
                cur_path,
                f"{args.data}/{idx}_{args.model}/output-{args.level}.json",
            )
            for idx in range(max_idx)
        ]  # sorted
        if args.num:
            folder_list = folder_list[: args.num]

        print(f"\nEvaluate {args.data} on {len(folder_list)} samples, {args.model}")
        print(f"level: {args.level}")

        scores_ratio = {r: [] for r in ratios}
        length_ratio = {r: [] for r in ratios}
        full_cache_mode = None
        for i, file in enumerate(folder_list):
            with open(file, "r") as f:
                data = json.load(f)

            preds = defaultdict(list)
            answers = []
            file_full_cache_mode = None
            task_names = [k for k in list(data.keys()) if k.startswith(args.task)]

            # parse generated responses from json files
            for fmt in task_names:
                for output_per_ratio in data[fmt]:
                    info, text = output_per_ratio
                    ratio_ = info[0]
                    preds[ratio_].append(text["pruned"])

                full_answer = text.get("full__")
                has_full_answer = full_answer is not None
                if (
                    file_full_cache_mode is not None
                    and file_full_cache_mode != has_full_answer
                ):
                    raise ValueError(f"mixed full-cache answers in {file}")
                file_full_cache_mode = has_full_answer
                if full_answer is not None and len(preds[1.0]) < len(
                    preds[ratios[-1]]
                ):
                    preds[1.0].append(full_answer)
                answers.append(text["answer"])

            if full_cache_mode is None:
                full_cache_mode = file_full_cache_mode
            elif file_full_cache_mode != full_cache_mode:
                raise ValueError(
                    "result directory mixes files with and without full-cache answers"
                )

            # for some tasks, evaluation require additional information (e.g., code language in repoqa)
            if answers_supp:
                answers = answers_supp[i]
            subtask = None
            if subtasks:
                subtask = subtasks[i]

            for r in ratios:
                if not preds[r]:
                    continue
                perf = _evaluate_answer(
                    preds[r], answers, args.data, args.task, subtask=subtask
                )
                scores_ratio[r].append(perf)

        print("avg_performance per ratio")
        perf_full = (
            avg_list_of_list(scores_ratio[1.0]) if scores_ratio[1.0] else None
        )
        for r in ratios:
            if not scores_ratio[r]:
                print("N/A")
                continue
            perf = avg_list_of_list(scores_ratio[r])
            print(f"{perf*100:.2f}")

            if perf_full:
                scores_ratio_all[r].append(perf / perf_full)

    print("=" * 50)
    print(data_list)
    print("Averaged relative performance (note, MRCR is not included)")
    for r in ratios:
        if scores_ratio_all[r]:
            print(f"{np.mean(scores_ratio_all[r]) * 100:.2f}")
        else:
            print("N/A")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.log_to_wandb and (args.wandb_project or args.wandb_entity):
        parser.error("--wandb-project and --wandb-entity require --log-to-wandb")
    if args.log_to_wandb and not args.run_dir:
        parser.error("--log-to-wandb requires --run-dir")
    if args.log_to_wandb and not args.wandb_project:
        parser.error("--log-to-wandb requires --wandb-project")
    if args.run_dir:
        _run_directory(args)
    else:
        _run_legacy(args)


if __name__ == "__main__":
    main()
