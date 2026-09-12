"""Golden cases from LongBench v2's pinned official pred.py answer extraction."""

from types import SimpleNamespace

import pytest
import torch

from results.evaluation_run import EvaluationRun
from results.metric import evaluate_answer
from results.parse import build_run_metrics
from utils.func import set_gen_length


@pytest.mark.parametrize(
    "prediction,answer,expected",
    [
        ("The correct answer is (A)", "A", 1),
        ("The correct answer is B.", "B", 1),
        ("**The correct answer is (C)**", "C", 1),
        ("Reasoning.\nThe correct answer is (D).", "D", 1),
        ("The correct answer is A then The correct answer is (B)", "B", 1),
        ("The correct answer is (A) then The correct answer is (B)", "B", 0),
        ("The correct answer is A then The correct answer is B", "A", 1),
        # Upstream intentionally has no word boundary after the bare letter.
        ("The correct answer is ABC", "A", 1),
        ("The correct answer is (B)", "A", 0),
        ("the correct answer is (A)", "A", 0),
        ("The correct answer is (a)", "A", 0),
        ("The correct answer is (E)", "A", 0),
        ("The correct answer is ( A )", "A", 0),
        ("The correct answer is: (A)", "A", 0),
        ("(A)", "A", 0),
        ("A", "A", 0),
        ("", "A", 0),
    ],
)
def test_official_answer_extraction_and_exact_accuracy(prediction, answer, expected):
    assert evaluate_answer([prediction], [[answer]], "longbench_v2", "qa") == [expected]


def test_v2_dispatch_preserves_explicit_similarity():
    assert evaluate_answer(
        ["one"], ["1"], "longbench_v2", "qa", similarity=True
    ) == [1]


def test_v2_cap_preserves_actual_greedy_generation_defaults(monkeypatch):
    from model import wrapper

    frozen_model = SimpleNamespace(
        name="unit", dtype=torch.float32, device=torch.device("cpu"), config=object()
    )
    monkeypatch.setattr(wrapper, "load_model", lambda name: (frozen_model, object()))
    monkeypatch.setattr(wrapper, "load_gate", lambda *args: None)
    monkeypatch.setattr(wrapper.ModelKVzip, "set_chat_template", lambda self: None)
    model = wrapper.ModelKVzip("unit")
    before = dict(model.gen_kwargs)
    assert before["do_sample"] is False
    assert set_gen_length("longbench_v2", model) == 128
    assert model.gen_kwargs == before | {"max_new_tokens": 128}
    assert set_gen_length("longbench_qasper", model) == 128
    assert set_gen_length("scbench_kv", model) == 96


def test_v2_online_and_saved_result_accuracy_match(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with EvaluationRun.open(
        tmp_path, "run", checkpoint_path=checkpoint, wandb_run_id=None,
        window_size=0, level="pair",
    ) as run:
        for index, prediction in enumerate(["The correct answer is (A)", "A"]):
            run.merge_example("longbench_v2", index, outputs={"qa": [[
                [0.2, 0.2, 0.0, 0.2],
                {"pruned": prediction, "full__": "The correct answer is (A)",
                 "answer": ["A"]},
            ]]})
        live = build_run_metrics(run, dataset_sizes={"longbench_v2": 2})
    offline = build_run_metrics(
        EvaluationRun.load(tmp_path / "run"), dataset_sizes={"longbench_v2": 2}
    )
    assert live == offline
    assert live["tasks"]["longbench_v2"]["ratios"]["0.2"]["score"] == 50
    assert live["tasks"]["longbench_v2"]["full_cache"]["score"] == 100
    assert live["tasks"]["longbench_v2"]["complete"]
