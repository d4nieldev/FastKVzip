"""Reconstruct the answer-training slice and audit long, contiguous text overlap.

No model or tokenizer is loaded. Downloading needs datasets/huggingface_hub;
rescanning an existing snapshot needs only Python's standard library.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import runpy


AGENTIC_REPO = "yzhuang/Agentic-Long-Context-Understanding-QA"
SCBENCH_REPO = "Jang-Hyun/SCBench-preprocessed"
PARSER = Path(__file__).parent / "data" / "load.py"
NORMALIZED = re.compile(r"</?para\s+\d+>|\w+", re.UNICODE)


def tokenize(text, mode):
    """Return tokens and their start/end character offsets in the original text."""
    if mode not in {"verbatim", "normalized"}:
        raise ValueError("mode must be verbatim or normalized")
    pattern = NORMALIZED if mode == "normalized" else re.compile(r"\S+")
    tokens, starts, ends = [], array("I"), array("I")
    for match in pattern.finditer(text):
        word = match.group()
        if mode == "normalized":
            if word.startswith("<"):
                continue
            word = word.casefold()
        tokens.append(word)
        starts.append(match.start())
        ends.append(match.end())
    return tokens, starts, ends


def _span(field, tokenized, start, end):
    _, starts, ends = tokenized
    char_start, char_end = starts[start], ends[end - 1]
    text = field["text"][char_start:char_end]
    return {
        **{key: value for key, value in field.items() if key != "text"},
        "word_start": start, "word_end": end,
        "char_start": char_start, "char_end": char_end,
        "excerpt": text if len(text) <= 600 else text[:300] + " … " + text[-300:],
    }


def find_overlaps(left_fields, right_fields, *, min_words=50, mode="normalized"):
    """Yield one representative maximal span per matching field pair.

    This detects every pair with an overlap >= min_words, but does NOT promise
    the globally longest occurrence or enumerate every occurrence in that pair.
    """
    if isinstance(min_words, bool) or not isinstance(min_words, int) or min_words < 1:
        raise ValueError("min_words must be a positive integer")
    if mode not in {"verbatim", "normalized"}:
        raise ValueError("mode must be verbatim or normalized")
    left_fields = list(left_fields)
    seed_size = max(1, min_words // 2)
    stride = min_words - seed_size + 1
    left_tokens = [tokenize(field["text"], mode) for field in left_fields]
    index = {}
    for field_index, (words, _, _) in enumerate(left_tokens):
        if len(words) < min_words:
            continue
        for start in range(0, len(words) - seed_size + 1, stride):
            seed = tuple(words[start:start + seed_size])
            index.setdefault(seed, {}).setdefault(field_index, []).append(start)

    # Every L-word overlap contains L-K+1 possible K-word seed starts. Sampling
    # that stride on the left and EVERY start on the right cannot miss a pair.
    # ponytail: repetitive subthreshold passages can cause quadratic candidate
    # expansion; no silent frequency/candidate cap that could hide contamination.
    for right in right_fields:
        right_tokens = tokenize(right["text"], mode)
        words, _, _ = right_tokens
        if len(words) < min_words:
            continue
        reported = set()
        for pos in range(len(words) - seed_size + 1):
            candidates = index.get(tuple(words[pos:pos + seed_size]), {})
            for field_index, starts in candidates.items():
                if field_index in reported:
                    continue
                other = left_tokens[field_index][0]
                for start in starts:
                    left_start, right_start = start, pos
                    while left_start and right_start and other[left_start - 1] == words[right_start - 1]:
                        left_start -= 1
                        right_start -= 1
                    left_end, right_end = start + seed_size, pos + seed_size
                    while left_end < len(other) and right_end < len(words) and other[left_end] == words[right_end]:
                        left_end += 1
                        right_end += 1
                    if left_end - left_start >= min_words:
                        reported.add(field_index)
                        yield {
                            "mode": mode, "word_count": left_end - left_start,
                            "left": _span(left_fields[field_index], left_tokens[field_index], left_start, left_end),
                            "right": _span(right, right_tokens, right_start, right_end),
                        }
                        break


def content_hash(row):
    """The deployed Agentic answer-cache content identity (not the cache key)."""
    text = json.dumps(
        {"context": row["context"], "question": row["question"][0]},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_adapters():
    # Import the existing concrete loaders without data/__init__.py, which also
    # imports the GPU/model runtime. Parsing and exact dedup stay canonical.
    return runpy.run_path(str(PARSER))


def reconstruct(args):
    from huggingface_hub import HfApi

    api = HfApi()
    agentic_revision = api.dataset_info(AGENTIC_REPO, revision=args.agentic_revision).sha
    scbench_revision = api.dataset_info(SCBENCH_REPO, revision=args.scbench_revision).sha
    adapters = load_adapters()
    print(f"Streaming Agentic at {agentic_revision}", flush=True)
    selected = adapters["load_agentic"](
        "train", teacher=None, answer_cache_dir=None,
        start=args.train_context_start, count=args.train_context_count,
        revision=agentic_revision,
    )
    if len(selected) != args.train_context_count:
        raise ValueError(f"requested {args.train_context_count} Agentic examples; found {len(selected)}")
    boundary = len(selected) - math.ceil(len(selected) * 0.1)
    agentic = [
        {**row, "index": args.train_context_start + i,
         "split": "train" if i < boundary else "validation",
         "content_sha256": content_hash(row)}
        for i, row in enumerate(selected)
    ]
    print(f"Loading {args.scbench_data} at {scbench_revision}", flush=True)
    benchmark = adapters["load_scbench"](args.scbench_data, revision=scbench_revision)
    end = args.scbench_start + args.scbench_count
    if end > len(benchmark):
        raise ValueError(f"requested SCBench range ends at {end}; dataset has {len(benchmark)} rows")
    scbench = [
        {**benchmark[i], "index": i, "split": "benchmark"}
        for i in range(args.scbench_start, end)
    ]
    provenance = {
        "agentic": {"repo": AGENTIC_REPO, "revision": agentic_revision,
                    "hf_split": "train", "start": args.train_context_start,
                    "count": args.train_context_count,
                    "deduplication": "exact-context-question-v1"},
        "scbench": {"repo": SCBENCH_REPO, "revision": scbench_revision,
                    "file": args.scbench_data + ".parquet", "hf_split": "train",
                    "start": args.scbench_start, "count": args.scbench_count},
        "loader_sha256": file_hash(PARSER),
        "historical_training_snapshot_verified": False,
        "limitation": "These are pinned audit snapshots, not independently verified historical training/evaluation revisions. Cache hashes verify membership only, not order or train/validation assignment.",
    }
    return agentic, scbench, provenance


def attach_cached_answers(rows, directories):
    """Read only: preserve matching archived answers AND their teacher identities."""
    by_hash = {row["content_sha256"]: row for row in rows}
    matched = set()
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"answer cache is not a directory: {directory}")
        for path in sorted(directory.glob("*.json")):
            with path.open() as handle:
                entry = json.load(handle)
            identity = entry.get("identity", {})
            if identity.get("dataset") != AGENTIC_REPO:
                continue
            row = by_hash.get(identity.get("content"))
            if row is None:
                continue
            if not isinstance(entry.get("answer"), str):
                raise ValueError(f"invalid answer in matching cache entry: {path}")
            row.setdefault("cached_answers", []).append({
                "answer": entry["answer"], "identity": identity,
                "cache_file": str(path.resolve()), "cache_sha256": file_hash(path),
            })
            matched.add(row["content_sha256"])
    return {
        "directories": [str(Path(path).resolve()) for path in directories],
        "matched_examples": len(matched), "selected_examples": len(rows),
        "note": "Archived answers retain their identities for review; matching content alone does not certify teacher settings or original row order. Missing answers are never regenerated.",
    }


def fields(rows, dataset):
    for row in rows:
        texts = [("context", row["context"])]
        texts += [(f"question:{i}", text) for i, text in enumerate(row["question"])]
        texts += [(f"answer:{i}", text) for i, text in enumerate(row["answers"] or [])]
        texts += [(f"cached_answer:{i}", item["answer"]) for i, item in enumerate(row.get("cached_answers", []))]
        for name, text in texts:
            yield {"id": f"{dataset}:{row['index']}:{name}", "split": row["split"],
                   "index": row["index"], "field": name, "text": text}


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_rows(path, rows):
    with Path(path).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_snapshot(directory):
    directory = Path(directory)
    with (directory / "manifest.json").open() as handle:
        manifest = json.load(handle)
    rows = []
    for name in ("agentic.jsonl", "scbench.jsonl"):
        path = directory / name
        if file_hash(path) != manifest["snapshot_sha256"][name]:
            raise ValueError(f"snapshot checksum mismatch: {path}")
        with path.open() as handle:
            rows.append([json.loads(line) for line in handle if line.strip()])
    return *rows, manifest["provenance"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing outputs are never overwritten.")
    parser.add_argument("--snapshot-dir", type=Path, help="Reuse an audit snapshot offline; do not download datasets.")
    parser.add_argument("--train-context-start", type=int, help="Default: 0.")
    parser.add_argument("--train-context-count", type=int, help="Default: 200, BEFORE the 10%% validation holdout.")
    parser.add_argument("--agentic-revision", help="Historical HF dataset commit if known; otherwise pin current revision for this audit.")
    parser.add_argument("--scbench-revision", help="Historical HF dataset commit if known; otherwise pin current revision for this audit.")
    parser.add_argument("--scbench-data", help="Default: scbench_kv.")
    parser.add_argument("--scbench-start", type=int, help="Default: 0.")
    parser.add_argument("--scbench-count", type=int, help="Default: 100.")
    parser.add_argument("--answer-cache-dir", type=Path, action="append", default=[], help="Optional local archived answer cache, read-only; repeat for both accounts.")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--mode", choices=("verbatim", "normalized", "both"), default="both")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    source_defaults = {
        "train_context_start": 0, "train_context_count": 200,
        "scbench_data": "scbench_kv", "scbench_start": 0, "scbench_count": 100,
        "agentic_revision": None, "scbench_revision": None,
    }
    if args.snapshot_dir and any(getattr(args, key) is not None for key in source_defaults):
        parser.error("dataset ranges/revisions cannot be changed when reusing a snapshot")
    for key, value in source_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.min_words < 1 or args.train_context_count < 2 or args.scbench_count < 1:
        parser.error("min-words/scbench-count must be positive; train-context-count must be at least 2")
    if args.train_context_start < 0 or args.scbench_start < 0:
        parser.error("range starts must be non-negative")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path")
    if args.snapshot_dir:
        agentic, scbench, provenance = read_snapshot(args.snapshot_dir)
    else:
        agentic, scbench, provenance = reconstruct(args)
    if args.answer_cache_dir:
        provenance = {**provenance, "answer_cache": attach_cached_answers(agentic, args.answer_cache_dir)}
    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "agentic.jsonl", agentic)
    write_rows(args.output_dir / "scbench.jsonl", scbench)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "snapshot_sha256": {name: file_hash(args.output_dir / name) for name in ("agentic.jsonl", "scbench.jsonl")},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(provenance["limitation"], flush=True)
    counts = Counter(row["split"] for row in agentic)
    context_sets = {
        split: {row["context"] for row in agentic if row["split"] == split}
        for split in ("train", "validation")
    }
    summary = {
        "min_words": args.min_words,
        "modes": ["verbatim", "normalized"] if args.mode == "both" else [args.mode],
        "agentic_examples": dict(counts), "scbench_contexts": len(scbench),
        "scbench_questions": sum(len(row["question"]) for row in scbench),
        "agentic_unique_raw_contexts": len({row["context"] for row in agentic}),
        "agentic_unique_raw_contexts_by_split": {split: len(values) for split, values in context_sets.items()},
        "train_validation_shared_raw_contexts": len(context_sets["train"] & context_sets["validation"]),
        "cached_answer_count": sum(len(row.get("cached_answers", [])) for row in agentic),
        "reported_pairs": {},
        "limitations": [
            provenance["limitation"],
            "One representative maximal span per field pair/mode, not the globally longest span or all occurrences; no estimate of total shared coverage.",
            "Verbatim matches preserve case/punctuation but ignore whitespace differences. Normalized matches use casefolded word characters and ignore paragraph tags/punctuation; review for boilerplate and identifier collisions.",
            "No-match means no contiguous sequence at this threshold in these inputs. It does not rule out short answer/key matches, paraphrases, prior model pretraining, or contamination outside the selected range.",
        ],
    }
    left, right = list(fields(agentic, "agentic")), list(fields(scbench, "scbench"))
    with (args.output_dir / "matches.jsonl").open("x") as handle:
        for mode in summary["modes"]:
            print(f"Scanning {mode}: {len(left)} Agentic fields × {len(right)} SCBench fields", flush=True)
            found = Counter()
            for match in find_overlaps(left, right, min_words=args.min_words, mode=mode):
                handle.write(json.dumps(match, ensure_ascii=False) + "\n")
                found[match["left"]["split"]] += 1
            summary["reported_pairs"][mode] = dict(found)
            print(f"{mode}: {dict(found)} matching field pairs", flush=True)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Audit complete: {args.output_dir.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    main()
