"""Complete-test-corpus inputs for reference-free summarization evaluation."""

from __future__ import annotations

import re

from data.benchmarks import BenchmarkDataset

SUMMARY_REQUEST = (
    "Summarize the supplied document in approximately 400–500 words. Cover its "
    "main developments, central ideas, and important conclusions. Base the summary "
    "on the supplied text."
)

GOVREPORT_DATASET = "ccdv/govreport-summarization"
GOVREPORT_REVISION = "4e21184e01ae8017e2c036e180fe5e541fef60a0"
PG19_DATASET = "emozilla/pg19"
PG19_REVISION = "b7bca68072ef1d86348f080bbda0996648d94315"

_BINS = (
    (8192, 16384, "[8192,16384)"),
    (16384, 32768, "[16384,32768)"),
    (32768, 65536, "[32768,65536)"),
    (65536, 131072, "[65536,131072)"),
)


def _book_id(sample):
    match = re.search(r"/ebooks/(\d+)(?:\D|$)", sample["url"])
    if match is None:
        raise ValueError(f"PG-19 row has no Gutenberg book ID: {sample['url']!r}")
    return int(match.group(1))


def _source_rows(name, loader):
    if name == "govreport_summary":
        samples = loader(
            GOVREPORT_DATASET,
            "document",
            split="test",
            revision=GOVREPORT_REVISION,
        )
        return [
            (
                f"govreport:test:{index:06d}",
                sample["report"],
                {"split": "test", "index": index},
            )
            for index, sample in enumerate(samples)
        ]
    if name == "pg19_summary":
        samples = loader(PG19_DATASET, split="test", revision=PG19_REVISION)
        rows = [
            (
                f"pg19:test:{_book_id(sample)}",
                sample["text"],
                {
                    "title": sample["short_book_title"],
                    "publication_date": sample["publication_date"],
                    "url": sample["url"],
                },
            )
            for sample in samples
        ]
        return sorted(rows, key=lambda row: int(row[0].rpartition(":")[2]))
    raise ValueError(f"Invalid summarization dataset: {name}")


def load_summarization(name, tokenizer, *, start=0, count=None, loader):
    if start < 0 or (count is not None and count < 0):
        raise ValueError("Summarization dataset range must be non-negative")

    source_rows = _source_rows(name, loader)
    bins = {label: 0 for _, _, label in _BINS}
    excluded = {"below_8192": 0, "at_least_131072": 0}
    excluded_documents = []
    eligible = []
    for source_id, context, source in source_rows:
        n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        label = next(
            (label for lower, upper, label in _BINS if lower <= n_tokens < upper),
            None,
        )
        if label is None:
            reason = (
                "below_8192" if n_tokens < _BINS[0][0] else "at_least_131072"
            )
            excluded[reason] += 1
            excluded_documents.append(
                {"id": source_id, "n_tokens": n_tokens, "reason": reason}
            )
            continue
        bins[label] += 1
        eligible.append(
            {
                "context": context,
                "question": [SUMMARY_REQUEST],
                "answers": [""],
                "task": "summarization",
                "id": source_id,
                "n_tokens": n_tokens,
                "length_bin": label,
                "source": source,
            }
        )

    inventory = {
        "source_size": len(source_rows),
        "eligible_size": len(eligible),
        "excluded": excluded,
        "excluded_documents": excluded_documents,
        "length_bins": bins,
    }
    stop = None if count is None else start + count
    return BenchmarkDataset(
        eligible[start:stop], full_size=len(eligible), inventory=inventory
    )
