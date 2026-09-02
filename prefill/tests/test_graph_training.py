import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graph import (
    GraphTrainer,
    ImplicitGraphScorer,
    PhaseTiming,
    TeacherExample,
    build_adamw_optimizers,
    build_scheduler,
    load_checkpoint,
    parse_scheduler_spec,
    save_checkpoint,
)
from graph.training import _model_gradient_norms


class Gate(nn.Module):
    def __init__(self, hidden=2, heads=1):
        super().__init__()
        self.nhead, self.ngroup, self.output_dim, self.sink = heads, 1, 1, 1
        self.d = 1.0
        self.q_proj = nn.Linear(hidden, heads, bias=True)
        self.k_proj = nn.Linear(hidden, heads, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.k_base = nn.Parameter(torch.ones(heads, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(heads, 1, 1))


def _config(layers=1, heads=1, hidden=2):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads,
        hidden_size=hidden,
    )


def _scorer(layers=1, heads=1):
    return ImplicitGraphScorer(
        [Gate(heads=heads).double() for _ in range(layers)],
        _config(layers, heads),
        graph_dim=2,
        graph_microbatch_size="auto",
        compute_dtype=torch.float64,
    )


def _example(layers=1, heads=1, tokens=5):
    return TeacherExample(
        dataset_name="unit",
        dataset_index=0,
        token_ids=torch.arange(tokens).view(1, -1),
        hidden_by_layer=[torch.randn(tokens, 2, dtype=torch.float64) for _ in range(layers)],
        teacher_scores=torch.rand(layers, 1, heads, tokens, dtype=torch.float64),
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        sequence_length=tokens,
    )


def _score_loss(scorer, example):
    hidden = torch.stack(tuple(example.hidden_by_layer))
    scores = scorer(hidden, token_microbatch_size=example.sequence_length)
    return torch.nn.functional.binary_cross_entropy(
        scores.squeeze(1), example.teacher_scores.squeeze(1), reduction="mean"
    )


def test_owned_teacher_example_shares_normal_cpu_capture_but_default_copies():
    hidden = torch.randn(3, 2)
    kwargs = dict(
        dataset_name="unit",
        dataset_index=0,
        token_ids=torch.arange(3).view(1, -1),
        hidden_by_layer=[hidden],
        teacher_scores=torch.rand(1, 1, 1, 3),
        prefix_ids=torch.tensor([[1]]),
        sequence_length=3,
    )
    copied = TeacherExample(**kwargs)
    owned = TeacherExample.from_owned_cpu(**kwargs)
    assert copied.hidden_by_layer[0].data_ptr() != hidden.data_ptr()
    assert owned.hidden_by_layer[0].data_ptr() == hidden.data_ptr()
    with torch.inference_mode():
        inference_hidden = torch.randn(3, 2)
    with pytest.raises(ValueError, match="normal CPU"):
        TeacherExample.from_owned_cpu(**{**kwargs, "hidden_by_layer": [inference_hidden]})


def test_scheduler_parsing_and_independent_specs():
    gate = parse_scheduler_spec("StepLR", '{"step_size": 1, "gamma": 0.5}')
    mixer = parse_scheduler_spec("ExponentialLR", '{"gamma": 0.9}')
    assert gate != mixer
    assert build_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))]), gate)
    with pytest.raises(ValueError, match="unknown"):
        parse_scheduler_spec("MadeUp")
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_scheduler_spec("LinearWarmupCosineLR", '{"warmup_fraction": 0}')


def test_linear_warmup_cosine_scheduler_uses_the_full_step_budget():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer,
        parse_scheduler_spec(
            "LinearWarmupCosineLR", '{"warmup_fraction": 0.5}'
        ),
        total_steps=10,
    )

    learning_rates = []
    for _ in range(10):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert learning_rates == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0, 0.9330127, 0.75, 0.5, 0.25, 0.0669873]
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_adamw_separates_mixer_weight_decay_groups_and_learning_rates():
    scorer = _scorer()
    gate, mixer = build_adamw_optimizers(
        scorer, gate_lr=2e-4, mixer_lr=3e-3, weight_decay=0.2
    )
    assert gate.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert mixer.param_groups[0]["lr"] == pytest.approx(3e-3)
    assert [group["weight_decay"] for group in mixer.param_groups] == [0.2, 0.0]
    assert set(mixer.param_groups[0]["params"]) == {
        scorer.mixer.in_proj.weight,
        scorer.mixer.out_proj.weight,
    }
    assert set(mixer.param_groups[1]["params"]) == {
        scorer.mixer.alpha,
        scorer.mixer.gamma,
        scorer.mixer.beta,
    }


def test_streamed_float64_gradient_matches_full_autograd():
    torch.manual_seed(4)
    reference = _scorer()
    with torch.no_grad():
        reference.mixer.gamma.copy_(torch.tensor([[1.3, 0.7]], dtype=torch.float64))
        reference.mixer.beta.copy_(torch.tensor([[0.2, -0.4]], dtype=torch.float64))
        reference.mixer.alpha.fill_(0.4)
    streamed = copy.deepcopy(reference)
    example = _example(tokens=5)
    for parameter in reference.gates.parameters():
        parameter.requires_grad_(False)
    for parameter in streamed.gates.parameters():
        parameter.requires_grad_(False)
    reference_loss = _score_loss(reference, example)
    reference_loss.backward()
    optimizer = torch.optim.SGD(streamed.mixer.parameters(), lr=0.0)
    trainer = GraphTrainer(streamed, mixer_optimizer=optimizer, token_microbatch_size=2)
    staged = trainer.train_mixer_phase(example)
    torch.testing.assert_close(staged.loss, reference_loss.detach(), rtol=1e-10, atol=1e-10)
    for (_, expected), (_, actual) in zip(
        reference.mixer.named_parameters(), streamed.mixer.named_parameters()
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-10, atol=2e-10)


def test_token_microbatch_invariance_and_one_mixer_step_per_context():
    torch.manual_seed(5)
    source = _scorer()
    first, second = copy.deepcopy(source), copy.deepcopy(source)
    example = _example(tokens=6)
    for scorer in (first, second):
        for parameter in scorer.gates.parameters():
            parameter.requires_grad_(False)
    a = GraphTrainer(first, mixer_optimizer=torch.optim.SGD(first.mixer.parameters(), lr=0))
    b = GraphTrainer(
        second,
        mixer_optimizer=torch.optim.SGD(second.mixer.parameters(), lr=0),
        token_microbatch_size=2,
    )
    a_result = a.train_mixer_phase(example)
    b_result = b.train_mixer_phase(example)
    assert a_result.optimizer_steps == b_result.optimizer_steps == 1
    torch.testing.assert_close(a_result.loss, b_result.loss, rtol=1e-10, atol=1e-10)
    for (_, left), (_, right) in zip(first.mixer.named_parameters(), second.mixer.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("compute_dtype", (torch.float16, torch.bfloat16))
def test_low_precision_staged_mixer_training(compute_dtype):
    scorer = ImplicitGraphScorer(
        [Gate().to(compute_dtype)],
        _config(),
        graph_dim=2,
        graph_microbatch_size="auto",
        compute_dtype=compute_dtype,
    )
    example = TeacherExample(
        dataset_name="unit",
        dataset_index=0,
        token_ids=torch.arange(5).view(1, -1),
        hidden_by_layer=[torch.randn(5, 2, dtype=compute_dtype)],
        teacher_scores=torch.rand(1, 1, 1, 5),
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        sequence_length=5,
    )
    trainer = GraphTrainer(
        scorer,
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0),
        token_microbatch_size=2,
    )
    result = trainer.train_mixer_phase(example)
    assert result.optimizer_steps == 1
    assert torch.isfinite(result.loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in scorer.mixer.parameters()
    )


def test_joint_context_updates_gate_and_mixer_once_with_independent_optimizers():
    scorer = _scorer()
    gate = torch.optim.SGD(scorer.gates.parameters(), lr=0.01)
    mixer = torch.optim.SGD(scorer.mixer.parameters(), lr=0.02)
    trainer = GraphTrainer(scorer, gate_optimizer=gate, mixer_optimizer=mixer, token_microbatch_size=2)
    result = trainer.train_context(_example(tokens=4), mode="joint")
    assert result["joint_loss"] is not None
    assert result["gate_steps"] == 1
    assert result["mixer_steps"] == 1
    gate_norms, mixer_norms = _model_gradient_norms(scorer)
    torch.testing.assert_close(result["gate_gradient_norm"], gate_norms.mean())
    torch.testing.assert_close(result["mixer_gradient_norm"], mixer_norms.mean())
    torch.testing.assert_close(
        result["gradient_norm"], torch.sqrt(gate_norms.square() + mixer_norms.square()).mean()
    )


def test_model_gradient_norms_average_per_layer_head_and_count_shared_gate_once():
    scorer = _scorer(heads=2)
    gate = scorer.gates[0]
    gate.q_norm = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    gate.q_proj.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    gate.q_norm.weight.grad = torch.tensor([[math.sqrt(8.0)]], dtype=torch.float64)
    scorer.mixer.alpha.grad = torch.tensor([12.0, 0.0], dtype=torch.float64)

    gate_norms, mixer_norms = _model_gradient_norms(scorer)

    torch.testing.assert_close(gate_norms, torch.tensor([math.sqrt(29.0), 2.0]))
    torch.testing.assert_close(mixer_norms, torch.tensor([12.0, 0.0]))


def test_two_phase_gate_steps_follow_token_microbatch_size():
    scorer = _scorer()
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0),
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0),
        token_microbatch_size=2,
    )
    result = trainer.train_context(_example(tokens=5), mode="two_phase")
    assert result["gate_steps"] == 3
    assert result["mixer_steps"] == 1


def test_checkpoint_round_trip_restores_current_mixer_optimizer_and_scheduler(tmp_path):
    scorer = _scorer()
    gate, mixer = build_adamw_optimizers(scorer)
    schedule = parse_scheduler_spec("LinearWarmupCosineLR", '{"warmup_fraction": 0.5}')
    gate_scheduler = build_scheduler(gate, schedule, total_steps=10)
    mixer_scheduler = build_scheduler(mixer, parse_scheduler_spec("ExponentialLR", '{"gamma": 0.9}'))
    gate.step()
    mixer.step()
    gate_scheduler.step()
    mixer_scheduler.step()
    config = {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": 1,
        "gate_sink": 1,
        "hidden_dim": 2,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 2,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 2,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
    }
    path = save_checkpoint(
        tmp_path, "last", scorer=scorer, config=config, model_id="unit",
        prefix_ids=torch.tensor([[1]]), prefill_chunk=2, data_cursor={"epoch": 0},
        wandb_run_id=None, gate_optimizer=gate, mixer_optimizer=mixer,
        gate_scheduler=gate_scheduler, mixer_scheduler=mixer_scheduler,
    )
    target = _scorer()
    target_gate, target_mixer = build_adamw_optimizers(target)
    restored_gate_scheduler = build_scheduler(target_gate, schedule, total_steps=10)
    restored_mixer_scheduler = build_scheduler(
        target_mixer, parse_scheduler_spec("ExponentialLR", '{"gamma": 0.9}')
    )
    payload = load_checkpoint(
        path, scorer=target, gate_optimizer=target_gate, mixer_optimizer=target_mixer,
        gate_scheduler=restored_gate_scheduler,
        mixer_scheduler=restored_mixer_scheduler,
    )
    assert payload["mixer_optimizer"] is not None
    assert restored_gate_scheduler.last_epoch == gate_scheduler.last_epoch
    assert restored_mixer_scheduler.last_epoch == mixer_scheduler.last_epoch
    for source, restored in zip(scorer.parameters(), target.parameters()):
        torch.testing.assert_close(source, restored)
    gate.step()
    target_gate.step()
    gate_scheduler.step()
    restored_gate_scheduler.step()
    assert target_gate.param_groups[0]["lr"] == pytest.approx(gate.param_groups[0]["lr"])


def test_phase_timing_accepts_joint_and_resolves_cpu_time():
    timing = PhaseTiming("cpu")
    with timing.region("joint", "forward"):
        pass
    assert timing.resolve()["joint_forward_seconds"] >= 0
