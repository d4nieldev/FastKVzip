import os
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from results.metric import evaluate_answer


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


if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os

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

    def retention_ratio(value):
        ratio = float(value)
        if not 0 < ratio < 1:
            raise argparse.ArgumentTypeError(
                "retention ratios must be between 0 and 1"
            )
        return ratio

    parser.add_argument("--ratios", nargs="+", type=retention_ratio)
    args = parser.parse_args()

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
                perf = evaluate_answer(
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
