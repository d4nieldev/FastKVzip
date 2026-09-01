import numpy as np

import data.load as data_load


def test_load_fineweb_uses_python_indices():
    class Samples:
        def __getitem__(self, index):
            assert type(index) is int
            return {"text": " context ", "token_count": 10_000}

    dataset, _, _ = data_load._select_fineweb_regular(
        Samples(), [np.int64(0)], count=1
    )
    assert dataset == {
        0: {"context": "context", "question": [""], "answers": [""]}
    }


def test_concatenation_does_not_skip_the_first_row_of_the_next_group():
    samples = [
        {"text": f"row-{index}", "token_count": token_count}
        for index, token_count in enumerate(
            [29_000, 29_000, 29_000, 20_000] * 2
        )
    ]

    dataset, _, last_source = data_load._select_fineweb_concat(
        samples, list(range(len(samples))), count=2
    )

    assert list(dataset) == [0, 4]
    assert dataset[0]["context"] == "\n\nrow-0\n\nrow-1\n\nrow-2\n\nrow-3"
    assert dataset[4]["context"] == "\n\nrow-4\n\nrow-5\n\nrow-6\n\nrow-7"
    assert last_source == 7


def test_training_split_uses_source_rows_and_starts_validation_after_both_pools(
    monkeypatch,
):
    samples = [
        {"text": f"row-{index}", "token_count": 5_000 if index == 1 else 20_000}
        for index in range(12)
    ]
    monkeypatch.setattr(
        data_load,
        "_load_fineweb_source",
        lambda: (samples, np.array([sample["token_count"] for sample in samples])),
    )

    selection = data_load.load_fineweb_training(train_context_count=2)

    assert selection.train_keys == (
        ("fineweb_10k", 0),
        ("fineweb_10k", 2),
        ("fineweb_10k_cat", 0),
    )
    assert selection.validation_keys == (
        ("fineweb_10k", 6),
        ("fineweb_10k", 7),
        ("fineweb_10k", 8),
        ("fineweb_10k_cat", 6),
    )
    assert selection.datasets["fineweb_10k_cat"][6]["context"].endswith(
        "\n\nrow-10"
    )
