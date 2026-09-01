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


def _scorer(layers=1, heads=1, *, graph_microbatch_size="auto", **mixer_options):
    return ImplicitGraphScorer(
        [Gate(heads=heads).double() for _ in range(layers)],
        _config(layers, heads),
        graph_dim=2,
        graph_microbatch_size=graph_microbatch_size,
        compute_dtype=torch.float64,
        **mixer_options,
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


def _score_loss(
    scorer,
    example,
    *,
    graph_microbatch_size=None,
    token_microbatch_size=None,
    rnf_seed=None,
):
    hidden = torch.stack(tuple(example.hidden_by_layer))
    scores = scorer(
        hidden,
        microbatch_size=graph_microbatch_size,
        token_microbatch_size=(
            example.sequence_length
            if token_microbatch_size is None
            else token_microbatch_size
        ),
        rnf_seed=rnf_seed,
    )
    return torch.nn.functional.binary_cross_entropy(
        scores.squeeze(1), example.teacher_scores.squeeze(1), reduction="mean"
    )


def _assert_gradients_close(expected, actual, *, rtol, atol):
    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_parameter in expected_parameters.items():
        actual_parameter = actual_parameters[name]
        assert expected_parameter.grad is not None, name
        assert actual_parameter.grad is not None, name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"{name}: {message}",
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


def test_adamw_granola_decay_membership_covers_each_mixer_parameter_once():
    scorer = _scorer(
        layers=2,
        heads=2,
        normalization="granola",
        normalization_sharing="global",
        granola_gnn_depth=2,
        granola_mlp_depth=2,
        granola_rnf_dim=3,
    )
    _, optimizer = build_adamw_optimizers(scorer, weight_decay=0.2)
    names_by_id = {
        id(parameter): name for name, parameter in scorer.mixer.named_parameters()
    }
    decay = {names_by_id[id(parameter)] for parameter in optimizer.param_groups[0]["params"]}
    no_decay = {
        names_by_id[id(parameter)] for parameter in optimizer.param_groups[1]["params"]
    }
    expected_decay = {"in_proj.weight", "out_proj.weight"} | {
        name
        for name in names_by_id.values()
        if name.startswith(
            ("granola_blocks.", "granola_gamma_head.", "granola_beta_head.")
        )
        and ".linears." in name
        and name.endswith(".weight")
    }

    assert decay == expected_decay
    assert no_decay == set(names_by_id.values()) - expected_decay
    assert decay.isdisjoint(no_decay)


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


@pytest.mark.parametrize("normalization", ("none", "granola"))
def test_streamed_gradients_match_full_autograd_for_new_normalizations(normalization):
    torch.manual_seed(14)
    granola = normalization == "granola"
    layers = heads = 2 if granola else 1
    mixer_options = {"normalization": normalization}
    if granola:
        mixer_options.update(
            normalization_sharing="global",
            granola_gnn_depth=2,
            granola_mlp_depth=2,
            granola_rnf_dim=3,
        )
    reference = _scorer(
        layers,
        heads,
        graph_microbatch_size=2 if granola else "auto",
        **mixer_options,
    )
    streamed = copy.deepcopy(reference)
    example = _example(layers, heads, tokens=5)
    for scorer in (reference, streamed):
        for parameter in scorer.gates.parameters():
            parameter.requires_grad_(False)

    # Both logical phases draw one RNF seed; resetting the generator makes the
    # full and exact-streamed paths consume the same draw without widening the
    # public trainer API.
    torch.manual_seed(97)
    reference_loss = _score_loss(reference, example)
    reference_loss.backward()
    trainer = GraphTrainer(
        streamed,
        mixer_optimizer=torch.optim.SGD(streamed.mixer.parameters(), lr=0.0),
        token_microbatch_size=2,
        graph_microbatch_size=2 if granola else "auto",
    )
    torch.manual_seed(97)
    staged = trainer.train_mixer_phase(example)

    torch.testing.assert_close(staged.loss, reference_loss.detach(), rtol=1e-10, atol=1e-10)
    _assert_gradients_close(
        reference.mixer,
        streamed.mixer,
        rtol=2e-8 if granola else 2e-10,
        atol=2e-9 if granola else 2e-10,
    )


def test_granola_explicit_rnf_seed_is_token_and_graph_microbatch_invariant():
    torch.manual_seed(15)
    source = _scorer(
        layers=2,
        heads=2,
        normalization="granola",
        normalization_sharing="global",
        granola_gnn_depth=2,
        granola_mlp_depth=2,
        granola_rnf_dim=3,
    )
    full, split = copy.deepcopy(source), copy.deepcopy(source)
    example = _example(layers=2, heads=2, tokens=6)
    for scorer in (full, split):
        for parameter in scorer.gates.parameters():
            parameter.requires_grad_(False)

    full_loss = _score_loss(
        full,
        example,
        graph_microbatch_size=4,
        token_microbatch_size=6,
        rnf_seed=73,
    )
    split_loss = _score_loss(
        split,
        example,
        graph_microbatch_size=1,
        token_microbatch_size=2,
        rnf_seed=73,
    )
    full_loss.backward()
    split_loss.backward()

    torch.testing.assert_close(split_loss, full_loss, rtol=2e-10, atol=2e-10)
    _assert_gradients_close(full.mixer, split.mixer, rtol=2e-8, atol=2e-9)


def test_streamed_granola_training_is_graph_microbatch_invariant():
    torch.manual_seed(16)
    source = _scorer(
        layers=2,
        heads=2,
        normalization="granola",
        normalization_sharing="global",
        granola_gnn_depth=2,
        granola_mlp_depth=2,
        granola_rnf_dim=3,
    )
    one, all_graphs = copy.deepcopy(source), copy.deepcopy(source)
    example = _example(layers=2, heads=2, tokens=5)
    for scorer in (one, all_graphs):
        for parameter in scorer.gates.parameters():
            parameter.requires_grad_(False)

    one_trainer = GraphTrainer(
        one,
        mixer_optimizer=torch.optim.SGD(one.mixer.parameters(), lr=0),
        token_microbatch_size=2,
        graph_microbatch_size=1,
    )
    all_trainer = GraphTrainer(
        all_graphs,
        mixer_optimizer=torch.optim.SGD(all_graphs.mixer.parameters(), lr=0),
        token_microbatch_size=2,
        graph_microbatch_size=4,
    )
    torch.manual_seed(101)
    one_result = one_trainer.train_mixer_phase(example)
    torch.manual_seed(101)
    all_result = all_trainer.train_mixer_phase(example)

    torch.testing.assert_close(one_result.loss, all_result.loss, rtol=2e-10, atol=2e-10)
    _assert_gradients_close(one.mixer, all_graphs.mixer, rtol=2e-8, atol=2e-9)


def test_granola_validation_draw_is_fixed_by_dataset_identity():
    scorer = _scorer(
        normalization="granola",
        granola_gnn_depth=2,
        granola_mlp_depth=2,
        granola_rnf_dim=3,
        normalization_seed=23,
    )
    trainer = GraphTrainer(scorer, token_microbatch_size=2)
    example = _example(tokens=5)
    other = TeacherExample(
        dataset_name=example.dataset_name,
        dataset_index=1,
        token_ids=example.token_ids,
        hidden_by_layer=example.hidden_by_layer,
        teacher_scores=example.teacher_scores,
        prefix_ids=example.prefix_ids,
        sequence_length=example.sequence_length,
    )

    first = trainer.evaluate_context(example).loss
    second = trainer.evaluate_context(example).loss
    different = trainer.evaluate_context(other).loss

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)


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
@pytest.mark.parametrize("normalization", ("none", "batchnorm", "granola"))
def test_low_precision_staged_mixer_training(compute_dtype, normalization):
    normalization_options = {"normalization": normalization}
    if normalization == "granola":
        normalization_options.update(
            granola_gnn_depth=2,
            granola_mlp_depth=2,
            granola_rnf_dim=3,
        )
    scorer = ImplicitGraphScorer(
        [Gate().to(compute_dtype)],
        _config(),
        graph_dim=2,
        graph_microbatch_size="auto",
        compute_dtype=compute_dtype,
        **normalization_options,
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
    gate_scheduler = build_scheduler(gate, parse_scheduler_spec("StepLR", '{"step_size": 1}'))
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
    restored_gate_scheduler = build_scheduler(
        target_gate, parse_scheduler_spec("StepLR", '{"step_size": 1}')
    )
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


@pytest.mark.parametrize(
    ("field", "mismatch"),
    (
        ("normalization", "batchnorm"),
        ("normalization_sharing", "global"),
        ("granola_gnn_depth", 3),
        ("granola_mlp_depth", 3),
        ("granola_rnf_dim", 4),
        ("normalization_seed", 8),
    ),
)
def test_direct_checkpoint_load_rejects_normalization_config_mismatch(
    tmp_path, field, mismatch
):
    options = {
        "normalization": "granola",
        "normalization_sharing": "layer",
        "granola_gnn_depth": 2,
        "granola_mlp_depth": 2,
        "granola_rnf_dim": 3,
        "normalization_seed": 7,
    }
    scorer = _scorer(layers=2, heads=2, **options)
    config = {
        "compute_dtype": "float64",
        "activation_order": "normalization-leaky-relu",
        **options,
    }
    config[field] = mismatch
    path = save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config=config,
        model_id="unit",
        prefix_ids=torch.tensor([[1]]),
        prefill_chunk=2,
        data_cursor={},
        wandb_run_id=None,
    )

    with pytest.raises(ValueError, match=field):
        load_checkpoint(path, scorer=scorer, restore_rng=False)


def test_phase_timing_accepts_joint_and_resolves_cpu_time():
    timing = PhaseTiming("cpu")
    with timing.region("joint", "forward"):
        pass
    assert timing.resolve()["joint_forward_seconds"] >= 0
