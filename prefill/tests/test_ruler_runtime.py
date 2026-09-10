from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig

from data.wrapper import DataWrapper
from model.wrapper import ModelKVzip
from utils.func import set_gen_length
from utils.tester import Evaluator


class Cache:
    def __init__(self):
        self.prefill_ids = torch.tensor([[1, 2, 3]])
        self._seen_tokens = 3

    def slice(self, length):
        self._seen_tokens = length


class Tokenizer:
    def decode(self, ids):
        return "".join(chr(token) for token in ids.tolist())


class Runtime(ModelKVzip):
    def __init__(self, generated, *, eos=0, generation_kwargs=None):
        self.tokenizer = Tokenizer()
        self.gen_kwargs = generation_kwargs or {}
        self.name = "qwen2.5-7b-instruct-1m"
        self.device = torch.device("cpu")
        self.generated_inputs = []

        def generate(input_ids, past_key_values, **_kwargs):
            self.generated_inputs.append(input_ids.clone())
            past_key_values._seen_tokens += input_ids.size(1) - 3 + len(generated) - 1
            return torch.cat(
                (input_ids, torch.tensor([generated], dtype=torch.long)), dim=1
            )

        self.model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=eos), generate=generate
        )

    def encode(self, text):
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)


def test_generation_keeps_the_final_answer_token_when_the_limit_is_reached():
    model = Runtime([ord("a"), ord("b")])
    cache = Cache()

    assert model.generate(torch.tensor([[4]]), kv=cache) == "ab"
    assert model.generated_inputs[0].tolist() == [[1, 2, 3, 4]]
    assert cache._seen_tokens == 3


@pytest.mark.parametrize(
    "generated,eos,kwargs,expected",
    [
        ([97, 0], 0, {}, "a"),
        ([97, 1], [0, 1], {}, "a"),
        ([0], 0, {}, ""),
        ([], 0, {}, ""),
        ([97, 98], 98, {"eos_token_id": 0}, "ab"),
        ([97, 0], 0, {"eos_token_id": None}, "a\x00"),
        ([97, 98], 0, {"eos_token_id": [98, 99]}, "a"),
        ([97, 98], None, {}, "ab"),
        ([97, 98], [], {}, "ab"),
        ([97, 98], 0, {"generation_config": GenerationConfig(eos_token_id=98)}, "a"),
        (
            [97, 98],
            0,
            {"generation_config": GenerationConfig(eos_token_id=98), "eos_token_id": 0},
            "ab",
        ),
        ([97, 0], 0, {"generation_config": None}, "a"),
        ([97, 0], 0, {"generation_config": GenerationConfig()}, "a"),
        (
            [97, 0],
            0,
            {"generation_config": GenerationConfig(), "use_model_defaults": False},
            "a",
        ),
        (
            [97, 0],
            0,
            {"generation_config": GenerationConfig(), "eos_token_id": None},
            "a\x00",
        ),
    ],
)
def test_generation_removes_only_the_effective_terminal_eos(
    generated, eos, kwargs, expected
):
    model = Runtime(generated, eos=eos, generation_kwargs=kwargs)

    assert model.generate(torch.tensor([[4]]), kv=Cache()) == expected


def test_generation_does_not_invent_an_eos_from_the_tokenizer():
    model = Runtime([97, 98], eos=None)
    model.tokenizer.eos_token_id = 98

    assert model.generate(torch.tensor([[4]]), kv=Cache()) == "ab"


@pytest.mark.parametrize("generated", [[97, 98], [97, 98, 0]])
def test_updated_cache_keeps_the_complete_non_eos_answer_in_prompt_history(generated):
    model = Runtime(generated)
    cache = Cache()

    assert model.generate(torch.tensor([[4]]), kv=cache, update_cache=True) == "ab"
    assert cache.prefill_ids.tolist() == [[1, 2, 3, 4, 97, 98]]


def test_ruler_preserves_multiple_references_for_one_question_in_results():
    model = Runtime([ord("a"), 0])
    wrapper = DataWrapper(
        "ruler_niah_single_1_4k",
        [
            {
                "context": "context",
                "question": ["find names"],
                "answers": [["alice", "bob"]],
            }
        ],
        model,
    )
    cache = Cache()

    inputs, info = wrapper.generate_answer(0, cache, prob=False)
    results = Evaluator(model, inputs, info)(cache)

    assert list(results) == ["qa"]
    assert results["qa"] == {"pruned": "a", "full__": "a", "answer": ["alice", "bob"]}


@pytest.mark.parametrize(
    "dataset,mode,expected_query,has_instruction",
    [
        ("ruler_niah_single_1_4k", "graphkv", "Q: find names", True),
        ("ruler_niah_single_1_4k", "official", "find names", False),
        ("scbench_kv", "official", "Q: find names", True),
    ],
)
def test_ruler_prompt_choice_keeps_existing_chat_boundaries_and_other_tasks(
    dataset, mode, expected_query, has_instruction
):
    model = Runtime([97, 0])
    wrapper = DataWrapper(
        dataset,
        [{"context": "context", "question": ["find names"], "answers": ["alice"]}],
        model,
        ruler_prompt_mode=mode,
    )

    inputs, info = wrapper.generate_answer(0, Cache(), prob=False)

    assert model.decode(inputs["qa"]["q"]) == (
        "\n\n" + expected_query + "<|im_end|>\n<|im_start|>assistant\n"
    )
    prefix = model.decode(model.sys_prompt_ids)
    assert prefix.startswith("<|im_start|>system\nYou are a helpful assistant.")
    assert ("Given the context" in prefix) is has_instruction
    assert Evaluator(model, inputs, info)(Cache())["qa"]["answer"] == "alice"


def test_ruler_rejects_an_unknown_prompt_choice():
    with pytest.raises(ValueError, match="ruler prompt mode"):
        DataWrapper(
            "ruler_niah_single_1_4k", [], Runtime([]), ruler_prompt_mode="invalid"
        )


@pytest.mark.parametrize(
    "task,expected",
    [
        ("niah_single_1", 128),
        ("niah_single_2", 128),
        ("niah_single_3", 128),
        ("niah_multikey_1", 128),
        ("niah_multikey_2", 128),
        ("niah_multikey_3", 128),
        ("niah_multivalue", 128),
        ("niah_multiquery", 128),
        ("vt", 30),
        ("cwe", 120),
        ("fwe", 50),
        ("qa_1", 32),
        ("qa_2", 32),
    ],
)
def test_ruler_generation_limits_are_selected_by_task(task, expected):
    model = Runtime([])

    assert set_gen_length(f"ruler_{task}_128k", model) == expected
    assert model.gen_kwargs["max_new_tokens"] == expected


def test_ruler_generation_limit_does_not_leak_to_the_next_dataset():
    model = Runtime([])
    set_gen_length("ruler_niah_single_1_4k", model)

    assert set_gen_length("scbench_kv", model) == 96
    assert model.gen_kwargs["max_new_tokens"] == 96
