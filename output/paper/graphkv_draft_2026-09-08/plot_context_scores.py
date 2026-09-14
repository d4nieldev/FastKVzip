"""Plot full-cache-relative scores by context length using matched examples.

Run with --recompute once after retrieving the completed run directories/logs.
Subsequent runs redraw the figure from the portable per-context score snapshot.
"""

import argparse
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import sys
from types import SimpleNamespace
from unittest.mock import patch

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[2]
WORKTREE = ROOT / ".worktrees/ruler-evaluation"
GRIDS = WORKTREE / ".slurm/grids"
DOWNLOAD = GRIDS / "baselines-core38/analysis-download"
EXTENSION = ROOT / ".slurm/grids/scbench-all-three-methods/analysis-download"
SNAPSHOT = PAPER / "data/context-analysis/context-scores.json"
RATIOS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
METHODS = ("graphkv", "fastkvzip", "kvzip")
LABELS = ("Graph-based selector", "FastKVzip", "KVzip")
EDGES = (0, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
BUCKET_LABELS = ("<4K", "4-8K", "8-16K", "16-32K", "32-64K", "64-128K", "128-256K")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bucket(tokens):
    if type(tokens) is not int or not 0 < tokens < EDGES[-1]:
        raise ValueError(f"Invalid/out-of-range context length: {tokens}")
    return bisect_right(EDGES, tokens) - 1


def summarize(rows):
    groups = [[] for _ in BUCKET_LABELS]
    for row in rows:
        groups[bucket(row["tokens"])].append(row)
    return [
        {"label": label, "lower_inclusive": EDGES[i], "upper_exclusive": EDGES[i + 1],
         "contexts": len(group), "questions": sum(r["questions"] for r in group),
         "task_counts": dict(sorted(Counter(r["task"] for r in group).items())),
         "scores": {method: [mean(r["scores"][method][j] for r in group) if group else None
                              for j in range(len(RATIOS))] for method in METHODS}}
        for i, (label, group) in enumerate(zip(BUCKET_LABELS, groups))
    ]


def summarize_relative(rows):
    """Normalize within task/length bucket before pooling with context weights.

    Each task's compressed mean is divided by its own method-specific full
    mean on the same contexts. Individual zero-score examples remain included.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[bucket(row["tokens"]), row["task"]].append(row)
    buckets = summarize(rows)
    for i, summary in enumerate(buckets):
        full_scores = {
            task: {method: mean(row["scores"][method][-1] for row in groups[i, task])
                   for method in METHODS}
            for task in summary["task_counts"]
        }
        for task, scores in full_scores.items():
            for method, score in scores.items():
                if not math.isfinite(score) or score <= 0:
                    raise ValueError(f"Undefined relative score: {task}, {summary['label']}, "
                                     f"{method} has full-cache mean {score}")
        summary["task_full_scores"] = full_scores
        summary["scores"] = {
            method: [
                mean(100 * row["scores"][method][j] / full_scores[task][method]
                     for task in summary["task_counts"] for row in groups[i, task])
                if summary["contexts"] else None
                for j in range(len(RATIOS))]
            for method in METHODS
        }
    return buckets


def logged_lengths(path, task):
    pattern = re.compile(r"# prefill \S+ " + re.escape(task) + r"-(\d+): (\d+) tokens")
    lengths = {}
    for index, tokens in pattern.findall(path.read_text()):
        assert index not in lengths or lengths[index] == int(tokens), (task, index)
        lengths[index] = int(tokens)
    return lengths


def supplementary_answers(paths):
    """Run the existing parser against pinned local files, without GPU imports."""
    import pyarrow.parquet as pq

    source = WORKTREE / "prefill/results/parse.py"
    parser = next(node for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "parse_answer")
    namespace = {"defaultdict": defaultdict}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(source), "exec"), namespace)

    def load_dataset(name, *, data_files, split):
        assert name == "Jang-Hyun/SCBench-preprocessed" and split == "train"
        return pq.read_table(paths[Path(data_files).stem]).to_pylist()

    with patch.dict(sys.modules, datasets=SimpleNamespace(load_dataset=load_dataset)):
        return {task: namespace["parse_answer"](task) for task in paths}


def recompute():
    # Import existing scoring/storage modules, not the GPU model/data loader.
    sys.path.insert(0, str(WORKTREE / "prefill"))
    from results.evaluation_run import EvaluationRun, atomic_write_json
    from results.metric import evaluate_answer
    source = read(PAPER / "data/method-scores.json")
    lengths = read(DOWNLOAD / "context-lengths.json")
    tasks = source["suite"]["tasks"]
    extension = read(EXTENSION / "source-manifest.json")
    old = read(SNAPSHOT)
    old_rows = {(row["task"], row["index"]): row for row in old["examples"]}
    extension_tasks = {run["data"] for run in extension["runs"]}
    assert len(tasks) == 60 and len(extension_tasks) == 22
    assert extension_tasks.isdisjoint(lengths) and set(tasks) == set(lengths) | extension_tasks
    repo_path = PAPER / "data/context-analysis/scbench_repoqa_short.parquet"
    supplementary_paths = {"scbench_repoqa_short": repo_path}
    extension_lengths = defaultdict(dict)
    for run in extension["runs"]:
        task = run["data"]
        if run["method"] in {"fastkvzip", "kvzip"}:
            observed = logged_lengths(Path(run["local_log"]), task)
            assert set(observed) == {str(i) for i in range(run["contexts"])}, (task, run["method"])
            extension_lengths[task][run["method"]] = observed
        if run.get("local_parquet") and any(tag in task for tag in ("many_shot", "repoqa", "summary_with_needles")):
            path = Path(run["local_parquet"])
            assert "a079be919d3131822c202180ffd4dc322968de45" in path.parts
            assert task not in supplementary_paths or digest(supplementary_paths[task]) == digest(path)
            supplementary_paths[task] = path
    for task, values in extension_lengths.items():
        assert set(values) == {"fastkvzip", "kvzip"} and values["fastkvzip"] == values["kvzip"], task
        lengths[task] = values["fastkvzip"]
    assert set(lengths) == set(tasks)
    expected_supplementary = {task for task in tasks if any(
        tag in task for tag in ("many_shot", "repoqa", "summary_with_needles"))}
    assert set(supplementary_paths) == expected_supplementary
    supplementary = supplementary_answers(supplementary_paths)
    locations = {method: {} for method in METHODS}
    graph_dir = GRIDS / "uniform-eval-core38/w002/production"
    manifests = list(graph_dir.glob("**/manifest.json")) + list(DOWNLOAD.glob("*/results/*/manifest.json"))
    manifests += [Path(run["local_run_dir"]) / "manifest.json" for run in extension["runs"]]
    run_ids = {v["run_id"]: k for k, v in source["methods"].items()}
    records, sources, max_error = {}, [], 0.0
    for manifest in manifests:
        run = EvaluationRun.load(manifest.parent)
        method = run_ids[run.manifest["wandb_run_id"]]
        identity = source["methods"][method]
        assert run.manifest["window_size"] == identity["window_size"]
        assert run.manifest["prefill_mode"] == identity["prefill_mode"]
        assert run.manifest["level"] == "pair"
        assert run.manifest["ruler_prompt_mode"] == "graphkv"
        assert run.manifest["generation_revision"] == 2 and run.manifest["window_revision"] == 1
        assert len(run.dataset_sizes) == 1
        task, count = next(iter(run.dataset_sizes.items()))
        assert task in tasks and task not in locations[method]
        locations[method][task] = str(run.run_dir)
        assert set(lengths[task]) == {str(i) for i in range(count)}
        if method != "graphkv":
            model = run.manifest["model_identity"]
            assert model["model_id"] == "Qwen/Qwen2.5-7B-Instruct-1M"
            assert model["model_revision"] == "e28526f7bb80e2a9c8af03b831a9af3812f18fba"
        local_scores, indices = [], set()
        for result in run.iter_examples():
            index = result.example_index
            assert index not in indices and result.task == task and result.has_full_answers
            indices.add(index)
            formats = sorted(result.formats, key=lambda f: 0 if f == "qa" else int(f.removeprefix("qa-")))
            assert formats == ["qa"] + [f"qa-{i}" for i in range(1, len(formats))]
            assert set(result.requested_ratios) == set(RATIOS[:-1])
            answers = [result.answers[fmt] for fmt in formats]
            key = (task, index)
            fingerprint = hashlib.sha256(json.dumps(answers, ensure_ascii=False).encode()).hexdigest()
            if key not in records:
                records[key] = {"task": task, "index": index, "tokens": lengths[task][str(index)],
                                "questions": len(formats), "reference_sha256": fingerprint, "scores": {}}
            row = records[key]
            assert row["questions"] == len(formats) and row["reference_sha256"] == fingerprint
            assert method not in row["scores"]
            references, subtasks = supplementary.get(task, ([], []))
            references = references[index] if references else answers
            subtask = subtasks[index] if subtasks else None
            if isinstance(references, dict):
                assert references["ground_truth"] == answers
            scores = []
            for ratio in RATIOS:
                if ratio == 1.0:
                    predictions = [result.full_answers[fmt] for fmt in formats]
                else:
                    predictions = [next(entry[1]["pruned"] for entry in result.payload[fmt]
                                        if float(entry[0][0]) == ratio) for fmt in formats]
                score = 100 * mean(evaluate_answer(predictions, references, task, "qa", subtask=subtask))
                assert math.isfinite(score) and 0 <= score <= 100
                scores.append(score)
            row["scores"][method] = scores
            local_scores.append(scores)
        assert indices == set(range(count))
        for j, ratio in enumerate(RATIOS):
            actual = mean(scores[j] for scores in local_scores)
            expected = identity["scores"][task][str(ratio)]
            error = abs(actual - expected)
            assert error < 1e-7, (method, task, ratio, actual, expected)
            max_error = max(max_error, error)
        sources.append({"method": method, "task": task, "run_dir": str(run.run_dir),
                        "manifest": run.manifest,
                        "outputs_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(
                            (run.run_dir / "outputs" / task).glob("*.json")))).hexdigest()})
        print(f"Verified {method}: {task} ({count} contexts)", flush=True)
    assert all(set(locations[m]) == set(tasks) for m in METHODS)
    rows = [records[key] for key in sorted(records)]
    assert len(rows) == 18531 and sum(r["questions"] for r in rows) == 28236
    assert all(set(r["scores"]) == set(METHODS) for r in rows)
    assert all(records[key] == row for key, row in old_rows.items()), "Previously reported per-context scores changed"
    # Dataset revision equality plus index/reference equality ties the three runs.
    for task in tasks:
        revisions = [s["manifest"]["dataset_revisions"] for s in sources if s["task"] == task]
        assert revisions[0] == revisions[1] == revisions[2]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(), "ratios": RATIOS,
        "weighting": "Each context example has equal weight after native evaluator aggregation within context, including the equal-component mean for RepoQA+KV. No task macro or retention-ratio averaging.",
        "length_definition": "Original raw context tokens from baseline prefill logs; excludes prefix, query, chat suffix and generation. K=1024; buckets lower-inclusive, upper-exclusive.",
        "coverage": {"tasks": len(tasks), "contexts": len(rows), "questions": sum(r["questions"] for r in rows)},
        "task_score_check": {"compared": len(tasks) * len(METHODS) * len(RATIOS),
                             "max_absolute_error": max_error, "tolerance": 1e-7},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [
            PAPER / "data/method-scores.json", DOWNLOAD / "context-lengths.json",
            DOWNLOAD / "source-manifest.json", EXTENSION / "source-manifest.json",
            *supplementary_paths.values(), WORKTREE / "prefill/results/parse.py",
            WORKTREE / "prefill/results/metric.py", WORKTREE / "prefill/results/repo_qa_utils.py"]},
        "sources": sources, "buckets": summarize(rows), "examples": rows,
    }
    atomic_write_json(SNAPSHOT, payload)
    return payload


def plot(data):
    os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "build/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    buckets = summarize_relative(data["examples"])
    ymax = max(110, 10 * math.ceil(max(
        value for b in buckets for scores in b["scores"].values()
        for value in scores if value is not None) / 10))
    fig, axes = plt.subplots(3, 2, figsize=(7, 8.4), sharex=True, sharey=True)
    colors, markers = ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^")
    for j, ax in enumerate(axes.flat):
        ax.axhline(100, color="#777777", linestyle="--", linewidth=0.9)
        for method, label, color, marker in zip(METHODS, LABELS, colors, markers):
            values = [b["scores"][method][j] if b["contexts"] else np.nan for b in buckets]
            ax.plot(range(len(buckets)), values, label=label, color=color, marker=marker,
                    linewidth=1.8, markersize=4.5)
        ax.set_title(f"{RATIOS[j]:.0%} retention" if RATIOS[j] < 1 else "Full cache (100%)", fontsize=11)
        ax.set_ylim(0, ymax + 3)
        ax.set_yticks(range(0, ymax + 1, 20))
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.6)
        ax.set_xticks(range(len(buckets)), [f"{b['label']}\nn={b['contexts']:,}" for b in buckets],
                      rotation=40, ha="right", fontsize=8.5)
        ax.tick_params(axis="x", labelbottom=True)
    fig.supylabel("Relative score (% of full cache)", x=0.01, fontsize=11)
    fig.supxlabel("Original context tokens per example (K = 1,024)", y=0.012, fontsize=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.53, 1.005), fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.94, hspace=0.9, wspace=0.16)
    destination = PAPER / "figures/context-score-buckets"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    relative = {
        "source": str(SNAPSHOT.relative_to(PAPER)), "source_sha256": digest(SNAPSHOT),
        "ratios": RATIOS, "coverage": data["coverage"],
        "normalization": "Within each task and length bucket, divide each method's compressed mean "
                         "by its own full-cache mean on the same contexts, multiply by 100, then "
                         "average task ratios weighted by context counts. Do not clip above 100. "
                         "A nonpositive full-cache task/bucket mean is an error, not an exclusion.",
        "buckets": buckets,
    }
    (SNAPSHOT.parent / "context-relative-scores.json").write_text(json.dumps(relative, indent=2) + "\n")
    print(json.dumps({"coverage": data["coverage"], "check": data["task_score_check"],
                      "buckets": [{k: v for k, v in b.items()
                                   if k not in {"task_counts", "task_full_scores"}} for b in buckets]}, indent=2))


def check():
    assert [bucket(n) for n in (1, 4095, 4096, 8191, 8192, 262143)] == [0, 0, 1, 1, 2, 6]
    rows = [{"task": "a", "tokens": 100, "questions": 10, "scores": {m: [0] * 6 for m in METHODS}},
            {"task": "a", "tokens": 150, "questions": 10, "scores": {m: [0] * 6 for m in METHODS}},
            {"task": "b", "tokens": 200, "questions": 1, "scores": {m: [100] * 6 for m in METHODS}}]
    bins = summarize(rows)
    assert bins[0]["scores"]["graphkv"] == [100 / 3] * 6  # not question/task weighting
    assert bins[0]["contexts"] == 3 and bins[0]["questions"] == 21
    assert bins[1]["scores"]["kvzip"] == [None] * 6
    # Two contexts from a harder task and one from an easier task. Normalize
    # before pooling: (2 * 50% + 100%) / 3, not the pooled ratio 125 / 150.
    relative_rows = [
        {"task": "a", "tokens": 100, "questions": 10,
         "scores": {m: [25] * 5 + [0] for m in METHODS}},
        {"task": "a", "tokens": 150, "questions": 10,
         "scores": {m: [0] * 5 + [50] for m in METHODS}},
        {"task": "b", "tokens": 200, "questions": 1,
         "scores": {m: [100] * 6 for m in METHODS}},
        {"task": "a", "tokens": 5000, "questions": 1,
         "scores": {m: [40] * 5 + [80] for m in METHODS}},
    ]
    relative_rows[2]["scores"]["kvzip"] = [100] * 5 + [50]
    relative = summarize_relative(relative_rows)
    assert math.isclose(relative[0]["scores"]["graphkv"][0], 200 / 3)
    assert math.isclose(relative[0]["scores"]["kvzip"][0], 100)  # own baseline; above-100 allowed
    assert relative[0]["contexts"] == 3 and relative[0]["questions"] == 21
    assert relative[1]["scores"]["graphkv"] == [50] * 5 + [100]  # bucket-specific baseline
    assert relative[2]["scores"]["graphkv"] == [None] * 6
    assert all(b["scores"][m][-1] == 100 for b in relative[:2] for m in METHODS)
    relative_rows[1]["scores"]["graphkv"][-1] = 0
    try:
        summarize_relative(relative_rows)
    except ValueError as error:
        assert "a, <4K, graphkv" in str(error)
    else:
        raise AssertionError("Accepted zero full-cache task/bucket mean")
    log = SimpleNamespace(read_text=lambda: "# prefill qwen scbench_kv-0: 100 tokens\n"
                          "# prefill qwen scbench_kv-0: 100 tokens\n"
                          "# prefill qwen other-0: 200 tokens\n")
    assert logged_lengths(log, "scbench_kv") == {"0": 100}
    log.read_text = lambda: "# prefill qwen scbench_kv-0: 100 tokens\n# prefill qwen scbench_kv-0: 101 tokens"
    try:
        logged_lengths(log, "scbench_kv")
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted conflicting context lengths")
    fixtures = {
        "scbench_many_shot": [{"prompts": ["context", "Pick\n(A) alpha\n(B) beta"], "ground_truth": ["B"]}],
        "scbench_repoqa_and_kv": [{"lang": "python", "repo": "r", "func_name": ["f", ""],
            "ground_truth": ["def f(): pass", "value"], "task": ["scbench_repoqa", "scbench_kv"]}],
        "scbench_summary_with_needles": [{"ground_truth": ["summary", "needle"],
            "task": ["scbench_summary", "scbench_kv"]}],
    }
    with patch("pyarrow.parquet.read_table", side_effect=lambda name: SimpleNamespace(to_pylist=lambda: fixtures[name])):
        parsed = supplementary_answers({name: name for name in fixtures})
    assert parsed["scbench_many_shot"] == ([["(B) beta"]], [])
    for task in ("scbench_repoqa_and_kv", "scbench_summary_with_needles"):
        assert parsed[task][1] == [fixtures[task][0]["task"]]
    assert parsed["scbench_repoqa_and_kv"][0][0]["func_name"] == ["f", ""]
    assert parsed["scbench_summary_with_needles"][0] == [["summary", "needle"]]
    print("PASS: bucket boundaries, raw/relative weighting, method/bucket baselines, zero baselines, "
          "empty buckets, log lengths, native supplementary parsing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check()
    if not args.check:
        plot(recompute() if args.recompute else read(SNAPSHOT))
