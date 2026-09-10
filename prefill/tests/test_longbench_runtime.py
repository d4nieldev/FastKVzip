from types import SimpleNamespace

import pytest
import torch

from data import DataWrapper
from model import ModelKVzip
from utils import Evaluator, set_gen_length


class Cache:
    """CPU cache implementing only the real prefill/evaluator boundary."""

    def __init__(self, evict_range):
        self.start_idx, self.end_idx = evict_range
        self.ctx_len = self.end_idx - self.start_idx
        self.sink = self.start_idx
        self.n_layers = self.n_heads_kv = 1
        self.hidden_cache = []
        self.key_cache = [torch.zeros(1)]
        self.valid = None
        self.protected_window = 0

    def _mem(self):
        return 0

    def init_score(self, get_score=False):
        self.score = [torch.zeros(1, 1, 0)]

    def prune(self, ratio, level):
        self.valid = torch.arange(self.ctx_len)[None, None, :] < int(ratio * self.ctx_len)
        return 0, self.valid.float().mean().item()

    def prune_chunk(self, ratio, evict_range, level):
        length = evict_range[1] - evict_range[0]
        valid = torch.arange(length)[None, None, :] < int(ratio * length)
        self.valid = valid if self.valid is None else torch.cat((self.valid, valid), -1)


class CPUModel(ModelKVzip):
    """Keep actual prefill and templates; replace only tokenization/LLM calls."""

    name = "unit"
    device = torch.device("cpu")

    def __init__(self):
        self.tokenizer = object()
        self.model = SimpleNamespace(_fastkvzip_revision="unit-revision")
        self.gates = object()
        self.kv_type = "retain"
        self.gen_kwargs = {"max_new_tokens": 512, "do_sample": False, "eos_token_id": [2, 3]}
        self.prefills, self.generations = [], []
        self.fail_prefill = False
        self.set_chat_template("unit")

    def set_chat_template(self, task):
        self.sys_prompt_ids = self.encode("chat prefix|")
        self.postfix_ids = self.encode("|close-user|assistant|")

    def encode(self, text):
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)

    def decode(self, ids):
        return "".join(chr(c) for c in ids.reshape(-1).tolist())

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        cache = Cache(evict_range)
        self.prefills.append(cache)
        return cache

    def __call__(self, input_ids, kv, **kwargs):
        if self.fail_prefill:
            raise RuntimeError("prefill failure")
        kv.score[0] = torch.cat((kv.score[0], torch.zeros(1, 1, input_ids.size(1))), -1)
        if kv.save_hidden:
            kv.hidden_cache.append(torch.zeros(1, input_ids.size(1), 2))

    def generate(self, query, *, kv):
        self.generations.append((query.clone(), kv, dict(self.gen_kwargs)))
        return "reference"


def row(**changes):
    return {
        "context": "def answer():\n    return 42\n",
        "context_prefix": "Please complete the code given below. \n",
        "question": ["\n  next_call()\nNext line of code:\n"],
        "answers": [["reference", "accepted alias"]],
    } | changes


@pytest.mark.parametrize("chunk_ratio", [1.0, 0.5, 0.2])
def test_only_raw_context_is_in_eviction_budget_and_prefix_does_not_accumulate(chunk_ratio):
    model = CPUModel()
    rows = [row(), row(context="another context", context_prefix="Different instruction:\n")]
    wrapper = DataWrapper("longbench_repobench-p", rows, model)
    # The graph evaluator restores this AFTER DataWrapper initializes templates.
    saved_prefix = model.encode("exact checkpoint prefix|")
    model.sys_prompt_ids = saved_prefix
    for index in [0, 1, 0]:
        cache = wrapper.prefill_context(index, chunk_ratio=chunk_ratio, window_size=0)
        expected = model.decode(saved_prefix) + rows[index]["context_prefix"]
        assert model.decode(cache.prefill_ids) == expected + rows[index]["context"]
        assert cache.start_idx == len(expected)
        assert cache.end_idx - cache.start_idx == len(rows[index]["context"])
        assert model.decode(cache.ctx_ids) == rows[index]["context"]
        assert model.sys_prompt_ids is saved_prefix
        if chunk_ratio < 1:
            assert cache.valid.size(-1) == len(rows[index]["context"])


def test_prefill_exception_restores_checkpoint_prefix():
    model = CPUModel()
    wrapper = DataWrapper("longbench_lcc", [row()], model)
    prefix = model.sys_prompt_ids
    model.fail_prefill = True
    with pytest.raises(RuntimeError, match="prefill failure"):
        wrapper.prefill_context(0)
    assert model.sys_prompt_ids is prefix


@pytest.mark.parametrize("question", ["\n  next_call()\nNext line of code:\n", ""])
@pytest.mark.parametrize("full_cache_answer", [True, False])
def test_raw_suffix_and_all_references_survive_real_evaluator(question, full_cache_answer):
    model = CPUModel()
    data = row(question=[question])
    wrapper = DataWrapper("longbench_repobench-p", [data], model)
    cache = wrapper.prefill_context(0)
    inputs, info = wrapper.generate_answer(0, cache, prob=False, full_cache_answer=full_cache_answer)
    assert model.decode(inputs["qa"]["q"]) == question + model.decode(model.postfix_ids)
    assert inputs["qa"]["gt"] == data["answers"][0]
    result = Evaluator(model, inputs, info)(cache)
    assert result == {"qa": {
        "pruned": "reference", "full__": "reference" if full_cache_answer else None,
        "answer": data["answers"][0],
    }}
    assert len(model.generations) == 1 + int(full_cache_answer)


def test_non_longbench_template_path_is_unchanged():
    model = CPUModel()
    data = row()
    wrapper = DataWrapper("scbench_kv", [data], model)
    prefix = model.sys_prompt_ids
    cache = wrapper.prefill_context(0)
    assert model.decode(cache.prefill_ids) == model.decode(prefix) + data["context"]
    inputs, _ = wrapper.generate_answer(0, cache, prob=False, full_cache_answer=False)
    assert model.decode(inputs["qa"]["q"]) == "\n\nQ: " + data["question"][0].rstrip() + model.decode(model.postfix_ids)


EXPECTED_CAPS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "multifieldqa_zh": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32, "dureader": 128,
    "gov_report": 512, "qmsum": 512, "multi_news": 512, "vcsum": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128, "lsht": 64,
    "passage_count": 32, "passage_retrieval_en": 32, "passage_retrieval_zh": 32,
    "lcc": 64, "repobench-p": 64,
}


@pytest.mark.parametrize("task,cap", EXPECTED_CAPS.items())
def test_official_generation_cap_is_the_only_generation_override(task, cap):
    model = CPUModel()
    before = dict(model.gen_kwargs)
    assert set_gen_length("longbench_" + task, model) == cap
    assert model.gen_kwargs == before | {"max_new_tokens": cap}


def test_caps_reset_when_switching_back_to_existing_benchmarks():
    model = CPUModel()
    set_gen_length("longbench_gov_report", model)
    assert set_gen_length("ruler_vt_4k", model) == 30
    assert model.gen_kwargs["max_new_tokens"] == 30


@pytest.mark.parametrize("protected_length", [16, 18])
def test_chunked_graph_prefill_waits_for_context_after_multiple_prefix_chunks(monkeypatch, protected_length):
    import graph.evaluation as graph_evaluation
    from test_graph_eval import _scorer

    class GraphModel(CPUModel):
        def __call__(self, input_ids, kv, **kwargs):
            # Real attention appends tokens within each layer until scoring clears it.
            hidden = torch.stack((input_ids.double(), torch.ones_like(input_ids)), -1)
            if kv.hidden_cache:
                kv.hidden_cache[0] = torch.cat((kv.hidden_cache[0], hidden), dim=1)
            else:
                kv.hidden_cache.append(hidden)

    model = GraphModel()
    model.gates = None  # The graph callback, not a gate forward, supplies scores.
    data = row(context="abcdefg", context_prefix="P" * (protected_length - 4))
    wrapper = DataWrapper("longbench_lcc", [data], model)
    prefix = model.encode("CKPT")
    model.sys_prompt_ids = prefix
    scorer = _scorer()
    scored_context, callbacks = [], []
    original_score_hidden = graph_evaluation.score_hidden_cache

    def score_hidden(scorer, hidden_cache, *, start_idx, end_idx, **kwargs):
        scored_context.append(hidden_cache[0][:, start_idx:end_idx, 0])
        return original_score_hidden(scorer, hidden_cache, start_idx=start_idx, end_idx=end_idx, **kwargs)

    monkeypatch.setattr(graph_evaluation, "score_hidden_cache", score_hidden)

    def score_chunk(cache):
        scores = graph_evaluation.score_context_chunk_cache(cache, scorer, token_microbatch_size=4)
        callbacks.append((scores, cache.score[0].size(-1),
                          cache.hidden_cache[0].size(1) if cache.hidden_cache else 0))

    cache = wrapper.prefill_context(0, prefill_chunk=8, chunk_ratio=0.5,
                                    window_size=0, chunk_scorer=score_chunk)
    # Both complete prefix chunks are deferred, including equality at the boundary.
    assert callbacks[:2] == [(None, 0, 8), (None, 0, 16)]
    torch.testing.assert_close(torch.cat(scored_context, dim=1), model.encode(data["context"]).double())
    assert cache.start_idx == protected_length
    assert cache.score[0].shape[-1] == protected_length + len(data["context"])
    assert not cache.score[0][..., :protected_length].any()
    expected_scores = torch.cat([scores[0] for scores, _, _ in callbacks if scores is not None], -1)
    torch.testing.assert_close(cache.score[0][..., protected_length:], expected_scores)
    assert cache.valid.tolist() == [[[True, True, True, False, False, False, False]]]
    assert cache.hidden_cache == []
    assert model.sys_prompt_ids is prefix
