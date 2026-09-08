import pytest

from data.ruler import parse_ruler_row


@pytest.mark.parametrize(
    ("task", "question"),
    [
        (
            "niah_single_1",
            "\nWhat is the special magic number for key mentioned in the provided text? The special magic number for key mentioned in the provided text is",
        ),
        (
            "niah_single_2",
            "\nWhat is the special magic number for key mentioned in the provided text? The special magic number for key mentioned in the provided text is",
        ),
        (
            "niah_single_3",
            "\nWhat is the special magic uuid for key mentioned in the provided text? The special magic uuid for key mentioned in the provided text is",
        ),
        (
            "niah_multikey_1",
            "\nWhat is the special magic number for key mentioned in the provided text? The special magic number for key mentioned in the provided text is",
        ),
        (
            "niah_multikey_2",
            "\nWhat is the special magic number for key mentioned in the provided text? The special magic number for key mentioned in the provided text is",
        ),
        (
            "niah_multikey_3",
            "\nWhat is the special magic uuid for key mentioned in the provided text? The special magic uuid for key mentioned in the provided text is",
        ),
        (
            "niah_multivalue",
            "\nWhat are all the special magic numbers for key mentioned in the provided text? The special magic numbers for key mentioned in the provided text are",
        ),
        (
            "niah_multiquery",
            "\nWhat are all the special magic numbers for key, and other mentioned in the provided text? The special magic numbers for key, and other mentioned in the provided text are",
        ),
        (
            "vt",
            "\nQuestion: Find all variables that are assigned the value 15311 in the text above.Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value 15311, they are: ",
        ),
        (
            "cwe",
            "\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:",
        ),
        (
            "fwe",
            "\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:",
        ),
        ("qa_1", "\nQuestion: In what country is Normandy located? Answer:"),
        (
            "qa_2",
            "\nQuestion: Were Scott Derrickson and Ed Wood of the same nationality? Answer:",
        ),
    ],
)
def test_published_prompt_families_preserve_demonstration_and_final_answer_cue(
    task, question
):
    # The earlier question belongs to the worked example, not the final query.
    context = (
        "Instructions and worked example" + question + " DEMO ANSWER\n\nActual context"
    )
    source = {"input": context + question, "outputs": ["first", "second"]}

    row = parse_ruler_row(task, source)

    assert row == {
        "context": context,
        "question": [question],
        "answers": [["first", "second"]],
    }
    assert row["context"] + row["question"][0] == source["input"]


@pytest.mark.parametrize(
    "source",
    [
        {"input": "no question", "outputs": ["target"]},
        {"input": "\nQuestion: no context", "outputs": ["target"]},
        {"input": "context\nQuestion: ", "outputs": ["target"]},
        {"input": "context\nQuestion: Q?", "outputs": "target"},
        {"input": "context\nQuestion: Q?", "outputs": []},
        {"input": "context\nQuestion: Q?", "outputs": [""]},
    ],
)
def test_malformed_ruler_rows_fail_clearly(source):
    with pytest.raises(ValueError, match="Malformed RULER"):
        parse_ruler_row("qa_1", source)


def test_ruler_loading_uses_pinned_task_cache_and_only_consumes_requested_range(
    monkeypatch,
):
    import datasets
    import huggingface_hub
    from data.load import load_dataset_all

    calls, consumed = [], []

    def source():
        for index in range(500):
            consumed.append(index)
            yield {
                "input": f"context {index}\nQuestion: Where? Answer:",
                "outputs": ["here", "there"],
            }

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return source()

    monkeypatch.setattr(datasets, "load_dataset", load)
    downloads = []

    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return "/cache/task.parquet"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    rows = load_dataset_all("ruler_qa_1_4k", None, start=2, count=1)

    assert downloads == [
        (
            ("lighteval/RULER-4096-Qwen2.5-Instruct",),
            {
                "filename": "data/qa_1-00000-of-00001.parquet",
                "repo_type": "dataset",
                "revision": "90daf679d2893abc90bbc9451f2a1f33de86c66e",
            },
        )
    ]
    assert calls == [
        (("parquet",), {"data_files": "/cache/task.parquet", "split": "train"})
    ]
    assert consumed == [0, 1, 2]
    assert len(rows) == 1
    assert rows.full_size == 500
    assert rows[0]["context"] == "context 2"
    assert rows[0]["answers"] == [["here", "there"]]
    assert len(load_dataset_all("ruler_qa_1_4k", None, n_data=None)) == 500
    assert len(load_dataset_all("ruler_qa_1_4k", None)) == 100
    assert len(load_dataset_all("ruler_qa_1_4k", None, count=0)) == 0
