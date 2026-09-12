import json

import pytest

from data.benchmarks import (
    RULER_EVALUATION_BENCHMARKS,
    dataset_revisions,
    get_data_list,
)
from data.longbench_v2 import (
    LONGBENCH_V2_PROTOCOL,
    LONGBENCH_V2_REVISION,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    load_longbench_v2,
    parse_longbench_v2_row,
)


def source_row(index=0):
    return {
        "_id": f"example-{index}",
        "context": f"\n  source {index}\n    code {{braces}} and $Q$\n\n",
        "question": "  Which choice contains {question} and $C_A$? \n",
        "choice_A": "  $C_B$ remains literal\n",
        "choice_B": "\n  second {choice}\n",
        "choice_C": "  third  ",
        "choice_D": "fourth\n",
        "answer": "B",
        "domain": "Code Repository Understanding",
    }


def test_v2_is_one_benchmark_without_extending_original_longbench_or_ruler_grid():
    names = get_data_list("all")
    assert len(names) == len(set(names)) == 129
    assert get_data_list("longbench_v2") == ["longbench_v2"]
    assert "longbench_v2" in names
    assert "longbench_v2" not in get_data_list("longbench")
    assert len(get_data_list("longbench")) == 21
    assert len(RULER_EVALUATION_BENCHMARKS) == 107
    assert "longbench_v2" not in RULER_EVALUATION_BENCHMARKS
    assert "agentic" not in names
    with pytest.raises(ValueError, match="LongBench"):
        get_data_list("longbench_v2_unknown")


def test_v2_revisions_are_separate_from_the_original_longbench_protocol():
    from data.longbench import LONGBENCH_PROTOCOL, LONGBENCH_REVISION

    expected = {
        "longbench_v2": LONGBENCH_V2_REVISION,
        "longbench_v2_protocol": LONGBENCH_V2_PROTOCOL,
    }
    assert dataset_revisions(["longbench_v2"]) == expected
    assert dataset_revisions(["longbench_v2", "longbench_qasper"]) == {
        **expected,
        "longbench": LONGBENCH_REVISION,
        "longbench_protocol": LONGBENCH_PROTOCOL,
    }
    assert dataset_revisions(["squad", "scbench_kv"]) == {}
    assert MAX_INPUT_TOKENS == 1_000_000
    assert MAX_NEW_TOKENS == 128


def test_official_direct_prompt_strips_only_source_boundaries_and_preserves_literals():
    source = source_row()
    row = parse_longbench_v2_row(source)
    assert row == {
        "context": "source 0\n    code {braces} and $Q$",
        "context_prefix": "Please read the following text and answer the question below.\n\n<text>\n",
        "question": [
            '\n</text>\n\nWhat is the correct answer to this question: Which choice contains {question} and $C_A$?\n'
            'Choices:\n(A) $C_B$ remains literal\n(B) second {choice}\n(C) third\n(D) fourth\n\n'
            'Format your response as follows: "The correct answer is (insert answer here)".'
        ],
        "answers": [["B"]],
    }
    assert source == source_row()


@pytest.mark.parametrize(
    "field,value",
    [("context", None), ("question", []), ("choice_A", None),
     ("choice_D", 2), ("answer", "a"), ("answer", "AB"), ("answer", None)],
)
def test_malformed_v2_rows_fail_clearly(field, value):
    source = source_row()
    source[field] = value
    with pytest.raises(ValueError, match="Malformed LongBench v2"):
        parse_longbench_v2_row(source)


@pytest.fixture
def published_json(monkeypatch, tmp_path):
    import huggingface_hub

    path = tmp_path / "data.json"
    rows = [source_row(index) for index in range(503)]
    rows[-1] = rows[-2].copy()  # Published duplicate rows must not disappear.
    path.write_text(json.dumps(rows), encoding="utf-8")
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_pinned_json_loader_keeps_full_benchmark_size_and_range_order(published_json):
    rows = load_longbench_v2(n_data=2, start=1)
    assert len(rows) == 2
    assert rows.full_size == 503
    assert rows == [parse_longbench_v2_row(source_row(i)) for i in (1, 2)]
    assert published_json == [
        (("zai-org/LongBench-v2",), {
            "filename": "data.json", "repo_type": "dataset",
            "revision": LONGBENCH_V2_REVISION,
        })
    ]
    full = load_longbench_v2()
    assert len(full) == full.full_size == 503
    assert full[-1] == full[-2]
    empty = load_longbench_v2(n_data=0)
    assert len(empty) == 0
    assert empty.full_size == 503
    assert len(load_longbench_v2(start=600)) == 0
    for kwargs in ({"n_data": -1}, {"start": -1}):
        with pytest.raises(ValueError, match="range"):
            load_longbench_v2(**kwargs)


def test_canonical_loader_maps_logical_test_to_published_data(published_json):
    from data.load import load_dataset_all

    rows = load_dataset_all("longbench_v2", None, start=3, count=1)
    assert len(rows) == 1
    assert rows.full_size == 503
    assert rows[0] == parse_longbench_v2_row(source_row(3))
    assert len(load_dataset_all("longbench_v2", None, n_data=None)) == 503
    with pytest.raises(ValueError, match="split"):
        load_dataset_all("longbench_v2", None, split="train")
