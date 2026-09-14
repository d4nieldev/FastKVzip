import random
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audit_data_overlap import find_overlaps, tokenize
import audit_data_overlap as audit


def _field(text, *, id="example", split="train", index=0, field="context"):
    return dict(id=id, split=split, index=index, field=field, text=text)


def _matches(left, right, *, min_words=3, mode="normalized"):
    return list(find_overlaps(left, right, min_words=min_words, mode=mode))


def _longest_shared_run(left, right):
    longest = 0
    for i in range(len(left)):
        for j in range(len(right)):
            length = 0
            while (
                i + length < len(left)
                and j + length < len(right)
                and left[i + length] == right[j + length]
            ):
                length += 1
            longest = max(longest, length)
    return longest


def _assert_offsets(match, left, right):
    for side, source in (("left", left), ("right", right)):
        result = match[side]
        tokens, starts, ends = tokenize(source["text"], match["mode"])
        assert all(result[key] == source[key] for key in ("id", "split", "index", "field"))
        assert result["word_end"] - result["word_start"] == match["word_count"]
        assert result["char_start"] == starts[result["word_start"]]
        assert result["char_end"] == ends[result["word_end"] - 1]
        excerpt = source["text"][result["char_start"] : result["char_end"]]
        assert result["excerpt"] == excerpt
        assert tokenize(excerpt, match["mode"])[0] == tokens[
            result["word_start"] : result["word_end"]
        ]
    assert tokenize(match["left"]["excerpt"], match["mode"])[0] == tokenize(
        match["right"]["excerpt"], match["mode"]
    )[0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("verbatim", ["<para", "12>", "Hello,", "Straße!", "</para", "12>", "NEXT_word."]),
        ("normalized", ["hello", "strasse", "next_word"]),
    ],
)
def test_tokenize_preserves_recoverable_character_offsets(mode, expected):
    raw = "  <para 12> Hello,\nStraße! </para 12>\tNEXT_word.  "
    tokens, starts, ends = tokenize(raw, mode)

    assert tokens == expected
    assert len(tokens) == len(starts) == len(ends)
    for token, start, end in zip(tokens, starts, ends):
        original = raw[start:end]
        assert (original if mode == "verbatim" else original.casefold()) == token


def test_normalization_ignores_paragraph_tags_but_verbatim_does_not():
    left = _field("lead <para 7> One, TWO </para 7> <para 8> three! </para 8> tail")
    right = _field("different one two three ending", id="scbench", split="test", index=9)

    matches = _matches([left], [right])

    assert len(matches) == 1
    assert matches[0]["word_count"] == 3
    assert matches[0]["mode"] == "normalized"
    assert matches[0]["left"]["excerpt"] == "One, TWO </para 7> <para 8> three"
    _assert_offsets(matches[0], left, right)
    assert _matches([left], [right], mode="verbatim") == []


def test_match_extends_to_both_ends_of_shared_sequence():
    shared = " ".join(f"word{i}" for i in range(27))
    left = _field(f"leftprefix {shared} leftsuffix")
    right = _field(f"rightprefix {shared} rightsuffix", id="scbench", split="test")

    matches = _matches([left], [right], min_words=8)

    assert len(matches) == 1
    assert matches[0]["word_count"] == 27
    assert matches[0]["left"]["word_start"] == 1
    assert matches[0]["left"]["word_end"] == 28
    _assert_offsets(matches[0], left, right)


@pytest.mark.parametrize("min_words", [1, 2, 3, 5, 8, 13])
def test_threshold_match_is_detected_at_every_anchor_alignment(min_words):
    shared = [f"shared{i}" for i in range(min_words)]
    for left_offset in range(min_words):
        for right_offset in range(min_words):
            left = _field(" ".join(["left"] * left_offset + shared + ["leftend"]))
            right = _field(" ".join(["right"] * right_offset + shared + ["rightend"]))
            matches = _matches([left], [right], min_words=min_words, mode="verbatim")
            assert len(matches) == 1, (min_words, left_offset, right_offset)
            assert matches[0]["word_count"] == min_words
            _assert_offsets(matches[0], left, right)


@pytest.mark.parametrize("mode", ["verbatim", "normalized"])
def test_random_small_sequences_agree_with_brute_force_existence(mode):
    rng = random.Random(42)
    for _ in range(150):
        left_words = rng.choices(["a", "b", "c", "d"], k=rng.randrange(16))
        right_words = rng.choices(["a", "b", "c", "d"], k=rng.randrange(16))
        threshold = rng.randrange(1, 8)
        left, right = _field(" ".join(left_words)), _field(" ".join(right_words))

        matches = _matches([left], [right], min_words=threshold, mode=mode)

        assert bool(matches) == (_longest_shared_run(left_words, right_words) >= threshold)
        assert len(matches) <= 1
        if matches:
            assert matches[0]["word_count"] >= threshold
            _assert_offsets(matches[0], left, right)


def test_below_threshold_and_separate_fields_cannot_form_a_match():
    assert _matches([_field("one two")], [_field("one two three")]) == []
    assert _matches(
        [_field("one two", field="context"), _field("three four", field="question")],
        [_field("one two three four")],
        min_words=4,
    ) == []


def test_repeated_tokens_produce_one_maximal_occurrence_per_pair():
    left = _field(" ".join(["same"] * 60))
    right = _field(" ".join(["same"] * 45))

    matches = _matches([left], [right], min_words=10)

    assert len(matches) == 1
    _assert_offsets(matches[0], left, right)
    result = matches[0]
    assert result["word_count"] >= 10
    assert result["left"]["word_start"] == 0 or result["right"]["word_start"] == 0
    assert result["left"]["word_end"] == 60 or result["right"]["word_end"] == 45


def test_each_matching_field_pair_is_reported_not_only_the_longest_pair():
    left = [
        _field("one two three four five", id="long", index=0),
        _field("six seven eight", id="short", index=1, field="question"),
    ]
    right = [_field("one two three four five stop six seven eight", id="scbench", split="test")]

    matches = _matches(left, right)

    assert len(matches) == 2
    assert {(row["left"]["id"], row["word_count"]) for row in matches} == {
        ("long", 5),
        ("short", 3),
    }


@pytest.mark.parametrize("left,right", [([], []), ([], [_field("a b c")]), ([_field("")], [_field("a b c")])])
def test_empty_inputs_produce_no_matches(left, right):
    assert _matches(left, right) == []


@pytest.mark.parametrize("min_words", [0, -1])
def test_minimum_overlap_must_be_positive(min_words):
    with pytest.raises(ValueError):
        _matches([_field("a b c")], [_field("a b c")], min_words=min_words)


def test_module_import_does_not_require_model_or_dataset_dependencies():
    script = """
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'datasets', 'numpy'}:
        raise AssertionError('Heavy dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import audit_data_overlap
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _snapshot(path):
    path.mkdir()
    agentic = [
        {"context": "one two three four", "question": [f"Question {i}?"], "answers": None,
         "split": split, "index": i}
        for i, split in enumerate(["train", "train", "validation"])
    ]
    for row in agentic:
        row["content_sha256"] = audit.content_hash(row)
    scbench = [{"context": "one two three four", "question": ["Q1", "Q2"],
                "answers": ["A1", "A2"], "split": "benchmark", "index": 0}]
    audit.write_rows(path / "agentic.jsonl", agentic)
    audit.write_rows(path / "scbench.jsonl", scbench)
    audit.write_json(path / "manifest.json", {
        "provenance": {"limitation": "Unverified historical revision"},
        "snapshot_sha256": {name: audit.file_hash(path / name) for name in ("agentic.jsonl", "scbench.jsonl")},
    })
    return agentic, scbench


def test_offline_cli_scans_both_splits_and_all_questions_without_download(tmp_path, monkeypatch):
    source, output = tmp_path / "snapshot", tmp_path / "scan"
    _snapshot(source)
    monkeypatch.setattr(audit, "reconstruct", lambda *_: pytest.fail("offline scan downloaded data"))

    summary = audit.main(["--snapshot-dir", str(source), "--output-dir", str(output), "--min-words", "4"])

    assert summary["agentic_examples"] == {"train": 2, "validation": 1}
    assert summary["scbench_questions"] == 2
    assert summary["agentic_unique_raw_contexts_by_split"] == {"train": 1, "validation": 1}
    assert summary["train_validation_shared_raw_contexts"] == 1
    assert summary["reported_pairs"] == {
        "verbatim": {"train": 2, "validation": 1},
        "normalized": {"train": 2, "validation": 1},
    }
    assert len((output / "matches.jsonl").read_text().splitlines()) == 6
    assert audit.read_snapshot(output)[0] == audit.read_snapshot(source)[0]
    with pytest.raises(SystemExit):
        audit.main(["--snapshot-dir", str(source), "--output-dir", str(output)])
    with pytest.raises(SystemExit):
        audit.main(["--snapshot-dir", str(source), "--output-dir", str(tmp_path / "other"), "--scbench-count", "1"])


def test_snapshot_checksum_change_is_an_error(tmp_path):
    source = tmp_path / "snapshot"
    _snapshot(source)
    with (source / "agentic.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.read_snapshot(source)


def test_answer_cache_membership_is_read_only_and_answers_keep_identity(tmp_path):
    source, cache = tmp_path / "snapshot", tmp_path / "cache"
    rows, _ = _snapshot(source)
    cache.mkdir()
    identity = {"dataset": audit.AGENTIC_REPO, "content": rows[0]["content_sha256"],
                "model": "recorded-teacher", "query": "Q: Question 0?"}
    path = cache / "entry.json"
    audit.write_json(path, {"identity": identity, "answer": "archived answer"})
    original = path.read_bytes()
    report = audit.attach_cached_answers(rows, [cache])
    assert report["matched_examples"] == 1
    assert report["selected_examples"] == 3
    assert rows[0]["cached_answers"][0]["identity"] == identity
    assert rows[0]["cached_answers"][0]["answer"] == "archived answer"
    assert "cached_answers" not in rows[-1]
    assert path.read_bytes() == original
    assert rows[0]["answers"] is None


def test_reconstruction_reuses_streaming_parser_dedup_and_pinned_loaders(tmp_path, monkeypatch):
    calls = []
    samples = [
        {"prompt": [{"role": "user", "content": content}]}
        for content in ["invalid", "doc\n\nQ0", "doc\n\nQ0", "doc\n\nQ1", "doc\n\nQ2", "doc\n\nQ3"]
    ]
    adapters = audit.load_adapters()

    def fake_load(repo, **kwargs):
        calls.append((repo, kwargs))
        if repo == audit.AGENTIC_REPO:
            def stream():
                yield from samples
                pytest.fail("stream read beyond requested deduplicated range")
            return stream()
        return [{"prompts": ["ctx", "first?", "second?"], "ground_truth": ["one", ["two", "three"]]}]

    adapters["load_agentic"].__globals__["load_dataset"] = fake_load
    monkeypatch.setattr(audit, "load_adapters", lambda: adapters)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda repo, **kw: SimpleNamespace(sha="a" * 40 if repo == audit.AGENTIC_REPO else "b" * 40))
    ))
    args = SimpleNamespace(agentic_revision=None, scbench_revision=None,
                           train_context_start=1, train_context_count=3,
                           scbench_start=0, scbench_count=1, scbench_data="scbench_kv")
    agentic, scbench, provenance = audit.reconstruct(args)
    assert [row["question"][0] for row in agentic] == ["Q1", "Q2", "Q3"]
    assert [row["split"] for row in agentic] == ["train", "train", "validation"]
    assert [row["index"] for row in agentic] == [1, 2, 3]
    assert all(row["answers"] is None for row in agentic)
    assert scbench[0]["question"] == ["first?", "second?"]
    assert scbench[0]["answers"] == ["one", "two, three"]
    assert provenance["historical_training_snapshot_verified"] is False
    assert calls == [
        (audit.AGENTIC_REPO, {"split": "train", "streaming": True, "revision": "a" * 40}),
        (audit.SCBENCH_REPO, {"data_files": "scbench_kv.parquet", "split": "train", "revision": "b" * 40}),
    ]
