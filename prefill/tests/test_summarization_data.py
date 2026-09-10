from types import SimpleNamespace

import pytest

import data.load as data_load
from data.summarization import SUMMARY_REQUEST


class LengthTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return range(int(text.removeprefix("tokens:")))


def test_govreport_inventory_bins_before_stable_slice(monkeypatch):
    samples = [
        {"report": "tokens:131072", "summary": "outside high"},
        {"report": "tokens:32768", "summary": "third"},
        {"report": "tokens:8191", "summary": "outside low"},
        {"report": "tokens:8192", "summary": "first"},
        {"report": "tokens:65536", "summary": "fourth"},
        {"report": "tokens:16384", "summary": "second"},
    ]
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(samples)

    monkeypatch.setattr(data_load, "load_dataset", load)

    rows = data_load.load_dataset_all(
        "govreport_summary", LengthTokenizer(), start=1, count=2
    )

    assert rows.full_size == 4
    assert rows.inventory == {
        "source_size": 6,
        "eligible_size": 4,
        "excluded": {"below_8192": 1, "at_least_131072": 1},
        "excluded_documents": [
            {
                "id": "govreport:test:000000",
                "n_tokens": 131072,
                "reason": "at_least_131072",
            },
            {
                "id": "govreport:test:000002",
                "n_tokens": 8191,
                "reason": "below_8192",
            },
        ],
        "length_bins": {
            "[8192,16384)": 1,
            "[16384,32768)": 1,
            "[32768,65536)": 1,
            "[65536,131072)": 1,
        },
    }
    assert [row["id"] for row in rows] == ["govreport:test:000003", "govreport:test:000004"]
    assert [row["n_tokens"] for row in rows] == [8192, 65536]
    assert rows[0]["question"] == [SUMMARY_REQUEST]
    assert rows[0]["answers"] == [""]
    assert rows[0]["task"] == "summarization"
    assert calls == [
        (
            ("ccdv/govreport-summarization", "document"),
            {
                "split": "test",
                "revision": "4e21184e01ae8017e2c036e180fe5e541fef60a0",
                "streaming": True,
            },
        )
    ]


def test_pg19_uses_complete_text_and_numeric_source_id_order(monkeypatch):
    samples = [
        {
            "short_book_title": "Ten",
            "publication_date": 1901,
            "url": "http://www.gutenberg.org/ebooks/10",
            "text": "tokens:16384",
        },
        {
            "short_book_title": "Two",
            "publication_date": 1888,
            "url": "http://www.gutenberg.org/ebooks/2",
            "text": "tokens:8192",
        },
    ]
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(samples)

    monkeypatch.setattr(data_load, "load_dataset", load)

    rows = data_load.load_dataset_all("pg19_summary", LengthTokenizer(), n_data=None)

    assert [row["id"] for row in rows] == ["pg19:test:2", "pg19:test:10"]
    assert rows[0]["context"] == "tokens:8192"
    assert rows[0]["source"] == {
        "title": "Two",
        "publication_date": 1888,
        "url": "http://www.gutenberg.org/ebooks/2",
    }
    assert calls == [
        (
            ("emozilla/pg19",),
            {
                "split": "test",
                "revision": "b7bca68072ef1d86348f080bbda0996648d94315",
                "streaming": True,
            },
        )
    ]


def test_summarization_loader_rejects_invalid_ranges(monkeypatch):
    monkeypatch.setattr(data_load, "load_dataset", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="range"):
        data_load.load_dataset_all("pg19_summary", LengthTokenizer(), start=-1)
