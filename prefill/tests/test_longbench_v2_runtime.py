from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from data import DataWrapper
from test_longbench_runtime import CPUModel, row


def setup(context, *, question="\nChoices: (A) first (B) second\n", capacity=6000):
    model = CPUModel()
    model.config = SimpleNamespace(max_position_embeddings=capacity)
    model.gen_kwargs["max_new_tokens"] = 128
    data = row(context=context, question=[question], answers=[["A"]])
    wrapper = DataWrapper("longbench_v2", [data], model)
    prefix = model.encode("saved token|")  # Same length as the native prefix, different text.
    model.sys_prompt_ids = prefix
    return model, data, wrapper, prefix


@pytest.mark.parametrize("chunk_ratio", [1.0, 0.5, 0.2])
def test_head_tail_context_keeps_prompts_and_reserves_native_kvzip_replay(chunk_ratio):
    context = "abcdefghij" * 800 + "END"
    model, data, wrapper, prefix = setup(context)
    original = deepcopy(data)
    caches = [wrapper.prefill_context(0, chunk_ratio=chunk_ratio, window_size=0) for _ in range(2)]
    for cache in caches:
        kept = cache.ctx_ids.shape[1]
        assert 3000 < kept < 4000  # KVzip needs ~2K replay tokens after prefill.
        front, back = (kept + 1) // 2, kept // 2
        assert model.decode(cache.ctx_ids) == context[:front] + context[-back:]
        assert model.decode(cache.prefill_ids) == model.decode(prefix) + data["context_prefix"] + model.decode(cache.ctx_ids)
        assert cache.ctx_len == kept  # Retention applies to the truncated context only.
        for _, repeat_ids in model.self_task(cache.ctx_ids):
            assert cache.prefill_ids.shape[1] + repeat_ids.shape[1] <= 6000
        assert cache.prefill_ids.shape[1] + len(data["question"][0]) + model.postfix_ids.shape[1] + 128 <= 6000
        if chunk_ratio < 1:
            assert cache.valid.shape[-1] == kept
    torch.testing.assert_close(caches[0].ctx_ids, caches[1].ctx_ids)
    assert data == original
    assert model.sys_prompt_ids is prefix


def test_long_question_and_options_determine_headroom_without_being_truncated():
    question = "\nQuestion and all four choices: " + "q" * 3500
    model, data, wrapper, prefix = setup("x" * 9000, question=question)
    cache = wrapper.prefill_context(0)
    assert cache.prefill_ids.shape[1] + len(question) + model.postfix_ids.shape[1] + 128 == 6000
    inputs, _ = wrapper.generate_answer(0, cache, prob=False, full_cache_answer=False)
    assert model.decode(inputs["qa"]["q"]) == question + model.decode(model.postfix_ids)
    assert model.sys_prompt_ids is prefix


def test_truncation_is_method_independent_and_short_rows_are_unchanged():
    contexts = []
    for gates in (None, object()):
        model, _, wrapper, _ = setup("x" * 9000)
        model.gates = gates
        # Scoring itself is exercised separately; the truncation must not branch on gates.
        model.prefill = lambda ids, **kw: SimpleNamespace(
            ctx_ids=ids, key_cache=[ids], _mem=lambda: 0,
        )
        contexts.append(wrapper.prefill_context(0, do_score=True).ctx_ids)
    torch.testing.assert_close(*contexts)
    model, _, wrapper, _ = setup("small context")
    assert model.decode(wrapper.prefill_context(0).ctx_ids) == "small context"


def test_shorter_checkpoint_prefix_does_not_give_graph_more_context_than_baselines():
    model, _, wrapper, _ = setup("x" * 9000)
    first = wrapper.prefill_context(0).ctx_ids
    model.sys_prompt_ids = model.encode("short|")
    second = wrapper.prefill_context(0).ctx_ids
    torch.testing.assert_close(first, second)


def test_longer_checkpoint_prefix_fails_instead_of_changing_only_graph_context():
    model, data, wrapper, _ = setup("x" * 9000, question="q" * 3500)
    prefix = model.encode("a much longer saved checkpoint prefix|")
    model.sys_prompt_ids = prefix
    with pytest.raises(ValueError, match="checkpoint prefix"):
        wrapper.prefill_context(0)
    assert model.prefills == []
    assert model.sys_prompt_ids is prefix
    data["context"] = "a short context"
    assert model.decode(wrapper.prefill_context(0).ctx_ids) == data["context"]


def test_one_million_total_budget_not_upstream_120k():
    model, data, wrapper, prefix = setup("x" * 1_020_000, capacity=1_010_000)
    cache = wrapper.prefill_context(0, prefill_chunk=1_100_000)
    assert 990_000 < cache.ctx_len < 1_000_000
    assert cache.prefill_ids.shape[1] + max(
        repeat.shape[1] for _, repeat in model.self_task(cache.ctx_ids[:, :4000])
    ) == 1_000_000


@pytest.mark.parametrize("capacity", [None, 0, True])
def test_unknown_or_invalid_capacity_fails_before_prefill_and_restores_prefix(capacity):
    model, _, wrapper, prefix = setup("context", capacity=capacity)
    with pytest.raises(ValueError, match="capacity"):
        wrapper.prefill_context(0)
    assert model.prefills == []
    assert model.sys_prompt_ids is prefix


def test_protected_prompt_that_cannot_fit_fails_without_dropping_question():
    model, _, wrapper, prefix = setup("context", question="q" * 6000)
    with pytest.raises(ValueError, match="prompt"):
        wrapper.prefill_context(0)
    assert model.prefills == []
    assert model.sys_prompt_ids is prefix


def test_one_token_budget_does_not_append_entire_context_via_negative_zero_slice():
    model, data, wrapper, prefix = setup("ABCDEFG", question="q" * 5800)
    model.config.max_position_embeddings = (
        prefix.shape[1] + len(data["context_prefix"]) + 5800 + model.postfix_ids.shape[1] + 128 + 1
    )
    cache = wrapper.prefill_context(0)
    assert model.decode(cache.ctx_ids) == "A"


def test_original_longbench_has_no_new_truncation_policy():
    model, data, _, _ = setup("x" * 9000)
    wrapper = DataWrapper("longbench_qasper", [data], model)
    assert wrapper.prefill_context(0).ctx_len == 9000
