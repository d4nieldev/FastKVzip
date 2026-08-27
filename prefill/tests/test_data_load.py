from types import SimpleNamespace

import data.load as data_load


def test_load_fineweb_uses_python_indices(monkeypatch):
    class Samples:
        data = SimpleNamespace(column=lambda name: [10_000])

        def __getitem__(self, index):
            assert type(index) is int
            return {"text": " context ", "token_count": 10_000}

    monkeypatch.setattr(data_load, "load_dataset", lambda *args, **kwargs: Samples())
    assert data_load.load_fineweb("fineweb_10k") == [
        {"context": "context", "question": [""], "answers": [""]}
    ]
