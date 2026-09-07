"""Adapt the pinned, pregenerated Qwen2.5-Instruct RULER datasets."""

from itertools import islice

from data.benchmarks import (
    BenchmarkDataset,
    RULER_LENGTHS,
    RULER_TASKS,
    parse_ruler_name,
)

# https://huggingface.co/datasets/lighteval/RULER-4096-Qwen2.5-Instruct
# Each published revision has all thirteen task splits, with 500 rows per split.
RULER_REVISIONS = {
    "4k": "90daf679d2893abc90bbc9451f2a1f33de86c66e",
    "8k": "e138066dd2bc6ded216ec8db21942ad03cb92f1a",
    "16k": "9228474efaef98ead1f4b3a032a195437dec0bc7",
    "32k": "1a76a1723fb1505d61988932b97b6ff0784d32ce",
    "64k": "7b35dc5e0a9bad6d748ec8240e00786644b66344",
    "128k": "5bdc2f0e2a6e2dc79abbc65378b504d400397415",
}
RULER_SAMPLES = 500


def parse_ruler_row(task, row):
    """Keep demonstrations in context and the final question/answer cue together."""
    if task not in RULER_TASKS:
        raise ValueError(f"Invalid RULER task: {task}")
    text, answers = row.get("input"), row.get("outputs")
    if (
        not isinstance(text, str)
        or not isinstance(answers, list)
        or not answers
        or not all(isinstance(answer, str) and answer.strip() for answer in answers)
    ):
        raise ValueError(
            f"Malformed RULER {task} row: expected input text and nonempty outputs"
        )
    if task in ("niah_multivalue", "niah_multiquery"):
        marker = "\nWhat are all the special magic "
    elif task.startswith("niah_"):
        marker = "\nWhat is the special magic "
    else:
        marker = "\nQuestion: "
    context, separator, question = text.rpartition(marker)
    if not separator or not context.strip() or not question.strip():
        raise ValueError(
            f"Malformed RULER {task} row: missing context or final question"
        )
    return {
        "context": context,
        "question": [separator + question],
        "answers": [answers],
    }


def load_ruler(name, n_data=None, *, start=0):
    """Read only the requested range through Hugging Face's standard streaming/cache."""
    from datasets import load_dataset

    task, length = parse_ruler_name(name)
    if start < 0 or (n_data is not None and n_data < 0):
        raise ValueError("RULER range must be non-negative")
    if n_data == 0:
        return BenchmarkDataset([], full_size=RULER_SAMPLES)
    samples = load_dataset(
        f"lighteval/RULER-{RULER_LENGTHS[length]}-Qwen2.5-Instruct",
        split=task,
        revision=RULER_REVISIONS[length],
        streaming=True,
    )
    stop = None if n_data is None else start + n_data
    return BenchmarkDataset(
        (parse_ruler_row(task, row) for row in islice(samples, start, stop)),
        full_size=RULER_SAMPLES,
    )
