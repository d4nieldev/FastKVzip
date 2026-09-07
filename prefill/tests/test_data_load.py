import sys
from types import ModuleType, SimpleNamespace

import pytest

import data.load as data_load
from data.wrapper import DataWrapper
import model.load as model_load


def test_squad_limit_counts_unique_contexts_and_keeps_all_their_questions(monkeypatch):
    samples = [
        {"context": context, "question": question, "answers": {"text": [answer]}}
        for context, question, answer in [
            ("A", "A1?", "A1"),
            ("A", "A2?", "A2"),
            ("B", "B1?", "B1"),
            ("C", "C1?", "C1"),
            ("B", "B2?", "B2"),
        ]
    ]
    monkeypatch.setattr(data_load, "load_dataset", lambda *_args, **_kwargs: samples)

    rows = data_load.load_dataset_all("squad", None, start=1, count=1)

    assert list(rows) == [
        {"context": "B", "question": ["B1?", "B2?"], "answers": ["B1", "B2"]}
    ]
    assert rows.full_size == 3
    assert len(data_load.load_dataset_all("squad", None, n_data=None)) == 3
    assert len(data_load.load_dataset_all("squad", None, count=0)) == 0


def test_gsm_range_applies_after_context_filtering_and_reports_full_size(monkeypatch):
    samples = [
        {"question": "too short. Skip?", "answer": "skip"},
        {"question": "context A. Ask A?", "answer": "answer A"},
        {"question": "context B. Ask B?", "answer": "answer B"},
        {"question": "context C. Ask C?", "answer": "answer C"},
    ]
    monkeypatch.setattr(data_load, "load_dataset", lambda *_args, **_kwargs: samples)
    tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: [1] * (1 if "short" in text else 72)
    )

    rows = data_load.load_dataset_all("gsm", tokenizer, start=1, count=1)

    assert len(rows) == 1
    assert rows[0]["question"] == ["Ask B?"]
    assert rows.full_size == 3
    assert len(data_load.load_dataset_all("gsm", tokenizer, n_data=None)) == 3
    assert len(data_load.load_dataset_all("gsm", tokenizer, count=0)) == 0
    assert samples[1]["question"] == "context A. Ask A?"


def test_scbench_limits_and_offsets_rows_without_changing_static_answers(monkeypatch):
    samples = [
        {"prompts": [f"context {i}", f"question {i}"], "ground_truth": [["a", "b"]]}
        for i in range(3)
    ]
    monkeypatch.setattr(data_load, "load_dataset", lambda *_args, **_kwargs: samples)

    rows = data_load.load_dataset_all("scbench_kv", None, start=1, count=1)

    assert list(rows) == [
        {"context": "context 1", "question": ["question 1"], "answers": ["a, b"]}
    ]
    assert rows.full_size == 3
    assert len(data_load.load_dataset_all("scbench_kv", None, n_data=None)) == 3
    assert len(data_load.load_dataset_all("scbench_kv", None, count=0)) == 0


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


def test_training_context_start_offsets_filtered_regular_and_concat_pools(monkeypatch):
    samples = Samples([5_000] + [20_000] * 15)
    monkeypatch.setattr(data_load, "load_dataset", lambda *args, **kwargs: samples)

    datasets, train_keys, validation_keys = data_load.load_fineweb_training(
        train_context_count=3,
        train_context_start=2,
    )

    assert train_keys == (
        ("fineweb_10k", 3),
        ("fineweb_10k", 4),
        ("fineweb_10k", 5),
        ("fineweb_10k_cat", 3),
    )
    assert validation_keys == (
        ("fineweb_10k", 8),
        ("fineweb_10k", 9),
        ("fineweb_10k", 10),
        ("fineweb_10k_cat", 8),
    )
    assert datasets["fineweb_10k_cat"][3]["context"].startswith("\n\nrow-3")
    assert datasets["fineweb_10k_cat"][3]["context"].endswith("\n\nrow-7")


def test_training_context_start_must_be_non_negative():
    with pytest.raises(ValueError, match="train context start must be non-negative"):
        data_load.load_fineweb_training(train_context_start=-1)


def _agentic_samples():
    return [
        {
            "prompt": [
                {"role": "system", "content": "ignore me"},
                {
                    "role": "user",
                    "content": "first paragraph\n\nsecond paragraph\n\nFirst question?",
                },
                {"role": "assistant", "content": "source answer to ignore"},
            ]
        },
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "first paragraph\n\nsecond paragraph\n\nFirst question?",
                },
                {"role": "assistant", "content": "duplicate source answer"},
            ]
        },
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "other paragraph\n\nSecond question?",
                }
            ]
        },
    ]


def test_agentic_loader_streams_parses_deduplicates_and_ranges_rows(monkeypatch):
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(_agentic_samples())

    monkeypatch.setattr(data_load, "load_dataset", load)

    dataset = data_load.load_dataset_all(
        "agentic", object(), split="test", start=1, count=1
    )

    assert data_load.available_splits("agentic") == frozenset({"train", "test"})
    assert calls == [
        (
            ("yzhuang/Agentic-Long-Context-Understanding-QA",),
            {"split": "test", "streaming": True},
        )
    ]
    assert len(dataset) == 1
    assert dataset[0] == {
        "context": "other paragraph",
        "question": ["Second question?"],
        "answers": None,
    }
    assert (
        len(
            data_load.load_dataset_all(
                "agentic", object(), split="test", start=1, count=0
            )
        )
        == 0
    )


class _AgenticTeacher:
    name = "unit-model"
    sys_prompt_ids = [11]
    postfix_ids = [12]
    gen_kwargs = {"max_new_tokens": 7, "do_sample": False}

    def __init__(
        self,
        *,
        model_id="org/unit-model",
        config_id=None,
        model_revision="a" * 40,
        tokenizer_id="org/unit-tokenizer",
        tokenizer_revision="b" * 40,
        template_token=13,
    ):
        self.model = SimpleNamespace(
            name_or_path=model_id,
            config=SimpleNamespace(
                _name_or_path=config_id or model_id,
                _commit_hash=model_revision,
            ),
        )
        self.tokenizer = SimpleNamespace(
            name_or_path=tokenizer_id,
            init_kwargs={"_commit_hash": tokenizer_revision},
        )
        self.template_token = template_token
        self.generated = 0

    def apply_template(self, query):
        return [self.template_token, query]

    def generate(self, query, *, kv):
        self.generated += 1
        return f"answer-{self.generated}"


def test_load_model_pins_runtime_objects_to_the_resolved_commit(monkeypatch):
    commit = "c" * 40
    config = SimpleNamespace(_commit_hash=commit)
    model = SimpleNamespace(dtype="unit", generation_config=SimpleNamespace())
    model.eval = lambda: None
    tokenizer = SimpleNamespace()
    model_calls, tokenizer_calls = [], []

    monkeypatch.setitem(
        sys.modules,
        "model.monkeypatch",
        ModuleType("model.monkeypatch"),
    )
    sys.modules["model.monkeypatch"].replace_attn = lambda *_args: None
    monkeypatch.setattr(
        model_load.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: config
    )
    monkeypatch.setattr(
        model_load.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model_calls.append((args, kwargs)) or model,
    )
    monkeypatch.setattr(
        model_load.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer_calls.append((args, kwargs)) or tokenizer,
    )

    loaded_model, loaded_tokenizer = model_load.load_model("first-org/unit-model")

    assert model_calls[0][1]["revision"] == commit
    assert tokenizer_calls[0][1]["revision"] == commit
    assert loaded_model._fastkvzip_canonical_id == "first-org/unit-model"
    assert loaded_model._fastkvzip_revision == commit
    assert loaded_tokenizer._fastkvzip_canonical_id == "first-org/unit-model"
    assert loaded_tokenizer._fastkvzip_revision == commit


def test_agentic_answers_are_lazy_and_reused_from_a_matching_cache(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    first_teacher = _AgenticTeacher()
    dataset = data_load.load_dataset_all(
        "agentic", object(), teacher=first_teacher, answer_cache_dir=tmp_path
    )

    assert dataset[0]["answers"] is None
    assert dataset.resolve_answers(0, object()) == ["answer-1"]
    assert dataset[0]["answers"] == ["answer-1"]
    assert first_teacher.generated == 1

    second_teacher = _AgenticTeacher()
    cached = data_load.load_dataset_all(
        "agentic", object(), teacher=second_teacher, answer_cache_dir=tmp_path
    )
    assert cached.resolve_answers(0, object()) == ["answer-1"]
    assert second_teacher.generated == 0


def test_agentic_cache_misses_for_distinct_full_model_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    first = _AgenticTeacher(model_id="first-org/unit-model")
    data_load.load_dataset_all(
        "agentic", object(), teacher=first, answer_cache_dir=tmp_path
    ).resolve_answers(0, object())
    second = _AgenticTeacher(model_id="second-org/unit-model")

    assert data_load.load_dataset_all(
        "agentic", object(), teacher=second, answer_cache_dir=tmp_path
    ).resolve_answers(0, object()) == ["answer-1"]
    assert second.generated == 1


def test_agentic_cache_prefers_runtime_full_ids_over_config_basenames(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    first = _AgenticTeacher(model_id="first-org/unit-model", config_id="short-model")
    data_load.load_dataset_all(
        "agentic", object(), teacher=first, answer_cache_dir=tmp_path
    ).resolve_answers(0, object())
    second = _AgenticTeacher(model_id="second-org/unit-model", config_id="short-model")

    assert data_load.load_dataset_all(
        "agentic", object(), teacher=second, answer_cache_dir=tmp_path
    ).resolve_answers(0, object()) == ["answer-1"]
    assert second.generated == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (("model_revision", "c" * 40), ("tokenizer_revision", "d" * 40)),
)
def test_agentic_cache_misses_when_a_component_revision_changes(
    monkeypatch, tmp_path, field, value
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    first = _AgenticTeacher()
    data_load.load_dataset_all(
        "agentic", object(), teacher=first, answer_cache_dir=tmp_path
    ).resolve_answers(0, object())
    second = _AgenticTeacher(**{field: value})

    assert data_load.load_dataset_all(
        "agentic", object(), teacher=second, answer_cache_dir=tmp_path
    ).resolve_answers(0, object()) == ["answer-1"]
    assert second.generated == 1


def test_agentic_cache_misses_when_applied_template_tokens_change(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    first = _AgenticTeacher()
    data_load.load_dataset_all(
        "agentic", object(), teacher=first, answer_cache_dir=tmp_path
    ).resolve_answers(0, object())
    second = _AgenticTeacher(template_token=14)

    assert data_load.load_dataset_all(
        "agentic", object(), teacher=second, answer_cache_dir=tmp_path
    ).resolve_answers(0, object()) == ["answer-1"]
    assert second.generated == 1


def test_agentic_persistent_cache_requires_immutable_teacher_revisions(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    dataset = data_load.load_dataset_all(
        "agentic",
        object(),
        teacher=_AgenticTeacher(model_revision=None),
        answer_cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="immutable model and tokenizer revisions"):
        dataset.resolve_answers(0, object())


def test_agentic_persistent_cache_rejects_branch_revisions(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    dataset = data_load.load_dataset_all(
        "agentic",
        object(),
        teacher=_AgenticTeacher(model_revision="main", tokenizer_revision="main"),
        answer_cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="immutable model and tokenizer revisions"):
        dataset.resolve_answers(0, object())


def test_agentic_cache_uses_loader_attached_tokenizer_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    teacher = _AgenticTeacher(tokenizer_revision=None)
    teacher.model._fastkvzip_canonical_id = teacher.model.name_or_path
    teacher.model._fastkvzip_revision = "a" * 40
    teacher.tokenizer._fastkvzip_canonical_id = teacher.tokenizer.name_or_path
    teacher.tokenizer._fastkvzip_revision = "b" * 40
    dataset = data_load.load_dataset_all(
        "agentic", object(), teacher=teacher, answer_cache_dir=tmp_path
    )

    assert dataset.resolve_answers(0, object()) == ["answer-1"]
    assert teacher.generated == 1


def test_agentic_answers_without_a_cache_are_resolved_per_call(monkeypatch):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    teacher = _AgenticTeacher()
    dataset = data_load.load_dataset_all("agentic", object(), teacher=teacher)

    assert dataset.resolve_answers(0, object()) == ["answer-1"]
    assert dataset[0]["answers"] is None
    assert dataset.resolve_answers(0, object()) == ["answer-2"]
    assert dataset[0]["answers"] is None
    assert teacher.generated == 2


def test_agentic_cache_mismatch_is_a_miss_and_matching_corruption_is_rejected(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        data_load, "load_dataset", lambda *_a, **_k: iter(_agentic_samples())
    )
    teacher = _AgenticTeacher()
    dataset = data_load.load_dataset_all(
        "agentic", object(), teacher=teacher, answer_cache_dir=tmp_path
    )
    dataset.resolve_answers(0, object())
    cache_file = next(tmp_path.glob("*.json"))
    entry = data_load.json.loads(cache_file.read_text())
    entry["identity"]["model"] = "old-model"
    cache_file.write_text(data_load.json.dumps(entry))

    mismatch = data_load.load_dataset_all(
        "agentic", object(), teacher=_AgenticTeacher(), answer_cache_dir=tmp_path
    )
    assert mismatch.resolve_answers(0, object()) == ["answer-1"]

    entry = data_load.json.loads(cache_file.read_text())
    entry["answer"] = 3
    cache_file.write_text(data_load.json.dumps(entry))
    corrupt = data_load.load_dataset_all(
        "agentic", object(), teacher=_AgenticTeacher(), answer_cache_dir=tmp_path
    )
    with pytest.raises(ValueError, match="corrupt Agentic answer cache"):
        corrupt.resolve_answers(0, object())


def test_wrapper_uses_deferred_full_answer_as_the_reference_without_regenerating():
    class Model:
        name = "unit"

        def __init__(self):
            self.generated = 0

        def set_chat_template(self, _task):
            pass

        def apply_template(self, query):
            return f"query:{query}"

        def encode(self, text):
            return f"ids:{text}"

        def generate(self, _query, *, kv):
            self.generated += 1
            return "unexpected duplicate"

    class DeferredDataset:
        def __init__(self):
            self.rows = [{"question": ["question"], "answers": None}]
            self.resolved = 0

        def __getitem__(self, index):
            return self.rows[index]

        def __len__(self):
            return len(self.rows)

        def resolve_answers(self, index, _full_kv):
            self.resolved += 1
            self.rows[index]["answers"] = ["full answer"]
            return self.rows[index]["answers"]

    model = Model()
    deferred = DeferredDataset()
    inputs, _info = DataWrapper("agentic", deferred, model).generate_answer(
        0, object(), prob=False
    )

    assert deferred.resolved == 1
    assert model.generated == 0
    assert inputs["qa"]["a"] == "ids:full answer"
    assert inputs["qa"]["gt"] == "ids:full answer"
