import json
import zipfile

import pytest

from data.benchmarks import (
    LONGBENCH_TASKS,
    RULER_EVALUATION_BENCHMARKS,
    dataset_revisions,
    get_data_list,
    parse_longbench_name,
)
from data.longbench import (
    LONGBENCH_REVISION,
    MAX_NEW_TOKENS,
    PROMPTS,
    load_longbench,
    parse_longbench_row,
)


def source_row(index=0):
    return {
        "context": f"\nDemonstration {{with braces}}\n  context {index}\n",
        "input": "  query {not formatting}\n",
        "answers": ["accepted answer", "another alias"],
        "all_classes": ["ABBR", "DESC"],
    }


def test_longbench_inventory_and_historical_grid_are_separate():
    names = get_data_list("longbench")
    assert len(names) == len(set(names)) == 21
    assert names == [f"longbench_{task}" for task in LONGBENCH_TASKS]
    assert set(PROMPTS) == set(MAX_NEW_TOKENS) == set(LONGBENCH_TASKS)
    assert len(get_data_list("all")) == len(set(get_data_list("all"))) == 129
    assert len(RULER_EVALUATION_BENCHMARKS) == 107
    assert not any(name.startswith("longbench_") for name in RULER_EVALUATION_BENCHMARKS)
    assert "agentic" not in get_data_list("all")
    assert get_data_list("longbench_repobench-p") == ["longbench_repobench-p"]
    assert parse_longbench_name("longbench_qasper") == "qasper"
    for name in ("longbench_unknown", "longbench_qasper_e", "qasper"):
        with pytest.raises(ValueError, match="LongBench"):
            parse_longbench_name(name)


def test_dataset_revisions_bind_new_protocol_without_changing_existing_benchmarks():
    from data.longbench import LONGBENCH_PROTOCOL
    from data.ruler import RULER_REVISIONS

    assert dataset_revisions(["squad", "scbench_kv"]) == {}
    previous = {"ruler_4k": RULER_REVISIONS["4k"]}
    assert dataset_revisions(["ruler_qa_1_4k", "ruler_vt_4k"]) == previous
    assert dataset_revisions(["ruler_qa_1_4k", "longbench_qasper"]) == {
        **previous, "longbench": LONGBENCH_REVISION,
        "longbench_protocol": LONGBENCH_PROTOCOL,
    }


@pytest.mark.parametrize("task", LONGBENCH_TASKS)
def test_all_task_templates_preserve_context_references_and_exact_formatting(task):
    source = source_row()
    row = parse_longbench_row(task, source)
    assert row["context"] == source["context"]
    assert row["answers"] == [source["answers"]]
    assert row["all_classes"] == source["all_classes"]
    assert len(row["question"]) == 1
    assert row["context_prefix"] + row["context"] + row["question"][0] == PROMPTS[task].format(**source)
    assert source["input"] not in row["context"]


def test_code_and_empty_input_suffixes_are_not_stripped():
    row = source_row()
    assert parse_longbench_row("repobench-p", row)["question"] == [
        row["input"] + "Next line of code:\n"
    ]
    row["input"] = ""
    assert parse_longbench_row("lcc", row)["question"] == ["Next line of code:\n"]
    assert parse_longbench_row("gov_report", row)["question"][0].endswith("Summary:")
    assert parse_longbench_row("passage_count", row)["question"][0].endswith(" ")


def test_official_null_classes_are_valid_except_on_classification_tasks():
    row = source_row()
    row["all_classes"] = None
    assert parse_longbench_row("narrativeqa", row)["all_classes"] == []
    for task in ("trec", "lsht"):
        with pytest.raises(ValueError, match="Malformed LongBench"):
            parse_longbench_row(task, row)


@pytest.mark.parametrize(
    "field,value",
    [("context", None), ("input", None), ("answers", []), ("answers", "answer"),
     ("answers", [None]), ("all_classes", "class")],
)
def test_malformed_source_rows_fail_clearly(field, value):
    row = source_row()
    row[field] = value
    with pytest.raises(ValueError, match="Malformed LongBench"):
        parse_longbench_row("qasper", row)


@pytest.fixture
def archive(monkeypatch, tmp_path):
    import huggingface_hub

    path = tmp_path / "data.zip"
    samples = [source_row(index) for index in range(4)]
    # Exact duplicate rows are separate published benchmark examples.
    samples.append(samples[-1])
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("data/qasper.jsonl", "\n".join(map(json.dumps, samples)))
        handle.writestr("data/unused.jsonl", "not JSON")
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_archive_is_pinned_and_only_selected_member_loaded(archive):
    rows = load_longbench("longbench_qasper", n_data=2, start=2)
    assert [row["context"] for row in rows] == [source_row(i)["context"] for i in (2, 3)]
    assert rows.full_size == 5
    assert archive == [
        (("zai-org/LongBench",), {
            "filename": "data.zip", "repo_type": "dataset",
            "revision": LONGBENCH_REVISION,
        })
    ]
    full = load_longbench("longbench_qasper")
    assert len(full) == full.full_size == 5
    assert full[-1] == full[-2]
    assert load_longbench("longbench_qasper", n_data=0).full_size == 5
    assert len(load_longbench("longbench_qasper", n_data=0)) == 0
    assert len(load_longbench("longbench_qasper", start=10)) == 0


def test_canonical_loader_range_and_split_contract(archive):
    from data.load import load_dataset_all

    rows = load_dataset_all("longbench_qasper", None, start=1, count=1)
    assert len(rows) == 1
    assert rows[0]["context"] == source_row(1)["context"]
    assert rows.full_size == 5
    assert len(load_dataset_all("longbench_qasper", None, n_data=None)) == 5
    with pytest.raises(ValueError, match="split"):
        load_dataset_all("longbench_qasper", None, split="train")
    for kwargs in ({"n_data": -1}, {"start": -1}):
        with pytest.raises(ValueError, match="range"):
            load_longbench("longbench_qasper", **kwargs)
