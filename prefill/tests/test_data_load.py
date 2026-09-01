from types import SimpleNamespace

import data.load as data_load


class Samples(list):
    def __init__(self, token_counts):
        super().__init__(
            {"text": f"row-{index}", "token_count": count}
            for index, count in enumerate(token_counts)
        )
        self.data = SimpleNamespace(
            column=lambda _name: [sample["token_count"] for sample in self]
        )

    def __getitem__(self, index):
        assert type(index) is int
        return super().__getitem__(index)


def test_load_fineweb_does_not_skip_concat_boundary_rows(monkeypatch):
    samples = Samples([29_000, 29_000, 29_000, 20_000] * 10)
    monkeypatch.setattr(data_load, "load_dataset", lambda *args, **kwargs: samples)

    dataset = data_load.load_fineweb("fineweb_10k_cat")

    assert len(dataset) == 10
    assert dataset[0]["context"].endswith("\n\nrow-3")
    assert dataset[1]["context"].startswith("\n\nrow-4")


def test_training_split_starts_validation_after_both_training_pools(monkeypatch):
    samples = Samples([20_000, 5_000] + [20_000] * 14)
    monkeypatch.setattr(data_load, "load_dataset", lambda *args, **kwargs: samples)

    datasets, train_keys, validation_keys = data_load.load_fineweb_training(6)

    assert train_keys == (
        ("fineweb_10k", 0),
        ("fineweb_10k", 2),
        ("fineweb_10k", 3),
        ("fineweb_10k", 4),
        ("fineweb_10k", 5),
        ("fineweb_10k", 6),
        ("fineweb_10k_cat", 0),
        ("fineweb_10k_cat", 6),
    )
    assert validation_keys == (
        ("fineweb_10k", 11),
        ("fineweb_10k", 12),
        ("fineweb_10k", 13),
        ("fineweb_10k_cat", 11),
    )
    assert datasets["fineweb_10k_cat"][6]["context"].startswith("\n\nrow-6")
    assert datasets["fineweb_10k_cat"][11]["context"].endswith("\n\nrow-15")
