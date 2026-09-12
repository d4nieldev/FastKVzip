"""Pinned LongBench v2 questions with the official direct-answer task prompt."""

import json

from data.benchmarks import BenchmarkDataset

LONGBENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONGBENCH_V2_PROTOCOL = "graphkv-direct-1m-v1"
MAX_INPUT_TOKENS = 1_000_000
MAX_NEW_TOKENS = 128

# MIT-licensed THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e,
# prompts/0shot.txt; full copyright and license notice: results/longbench.py.
PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def parse_longbench_v2_row(source):
    fields = ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D")
    if (
        not all(isinstance(source.get(field), str) for field in fields)
        or source.get("answer") not in ("A", "B", "C", "D")
    ):
        raise ValueError("Malformed LongBench v2 row: invalid text, choices or answer")
    values = {field: source[field].strip() for field in fields}
    prefix, suffix = PROMPT.split("{context}")
    return {
        "context": values["context"],
        "context_prefix": prefix,
        "question": [suffix.format(**values)],
        "answers": [[source["answer"]]],
    }


def load_longbench_v2(n_data=None, *, start=0):
    """Expose the published train-named evaluation file as our logical test split."""
    from huggingface_hub import hf_hub_download

    if start < 0 or (n_data is not None and n_data < 0):
        raise ValueError("LongBench v2 range must be non-negative")
    path = hf_hub_download(
        "zai-org/LongBench-v2", filename="data.json", repo_type="dataset",
        revision=LONGBENCH_V2_REVISION,
    )
    with open(path, encoding="utf-8") as handle:
        samples = json.load(handle)
    stop = None if n_data is None else start + n_data
    return BenchmarkDataset(
        [parse_longbench_v2_row(source) for source in samples[start:stop]],
        full_size=len(samples),
    )
