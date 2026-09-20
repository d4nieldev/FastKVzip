"""Gate-space coupling and implicit self loops.

Under `--mixer-coupling gate-space` the mixer's graph-width features reach the
gate through zero-initialized maps into its normalized query and key spaces and
its logit bias, instead of a hidden-width residual. `--self-loop-init` adds a
learnable `lambda I` to the implicit adjacency. Both must work with either
mixer architecture and every normalization, and the hidden default must stay
exactly what it was.
"""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from attention.gate import Weight
from graph import (
    ACTIVATION_ORDER,
    INJECTION_TARGETS,
    GateInjection,
    GraphTrainer,
    ImplicitGraphScorer,
    TeacherExample,
    build_adamw_optimizers,
    load_checkpoint,
    replay_score_gradients,
    save_checkpoint,
    score_context_subgraphs,
)
from graph.model import _HeadwiseGateAdapter

HIDDEN, GRAPH_DIM, GATE_DIM, GROUPS, SINK = 6, 4, 4, 2, 2
GPS = dict(mixer_architecture="gps", gps_attention_heads=2, gps_random_features=8)
GRANOLA = dict(
    normalization="granola",
    normalization_sharing="layer",
    granola_gnn_depth=2,
    granola_mlp_depth=2,
    granola_rnf_dim=3,
)


def _gate(layer: int, heads: int) -> Weight:
    """The released gate's own module, with real RMSNorms and sink keys."""

    gate = Weight(layer, HIDDEN, GATE_DIM, heads, GROUPS, torch.float64, sink=SINK)
    with torch.no_grad():
        nn.init.normal_(gate.k_base, std=0.5)
        nn.init.normal_(gate.b, std=0.1)
    return gate.double()


def _config(layers: int, heads: int):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads * GROUPS,
        hidden_size=HIDDEN,
    )


def _scorer(layers: int = 2, heads: int = 2, **options) -> ImplicitGraphScorer:
    options.setdefault("graph_dim", GRAPH_DIM)
    options.setdefault("graph_microbatch_size", "auto")
    return ImplicitGraphScorer(
        [_gate(layer, heads) for layer in range(layers)],
        _config(layers, heads),
        compute_dtype=torch.float64,
        **options,
    )


def _example(layers: int = 2, heads: int = 2, tokens: int = 9) -> TeacherExample:
    return TeacherExample(
        dataset_name="unit",
        dataset_index=0,
        token_ids=torch.arange(tokens).view(1, -1),
        hidden_by_layer=[
            torch.randn(tokens, HIDDEN, dtype=torch.float64) for _ in range(layers)
        ],
        teacher_scores=torch.rand(layers, 1, heads, tokens, dtype=torch.float64),
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        sequence_length=tokens,
    )


def _stacked(example: TeacherExample) -> torch.Tensor:
    return torch.stack(tuple(example.hidden_by_layer))


def _randomize_injection(scorer: ImplicitGraphScorer, std: float = 0.5) -> None:
    """Zero maps make every downstream check trivial; give them real values."""

    with torch.no_grad():
        for parameter in scorer.mixer.injection.parameters():
            parameter.normal_(std=std)


def _full_loss(scorer: ImplicitGraphScorer, example: TeacherExample) -> torch.Tensor:
    scores = scorer(_stacked(example), token_microbatch_size=example.sequence_length)
    return torch.nn.functional.binary_cross_entropy(
        scores.squeeze(1), example.teacher_scores.squeeze(1), reduction="mean"
    )


def _assert_gradients_close(expected: nn.Module, actual: nn.Module, *, rtol, atol) -> None:
    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_parameter in expected_parameters.items():
        assert expected_parameter.grad is not None, name
        assert actual_parameters[name].grad is not None, name
        torch.testing.assert_close(
            actual_parameters[name].grad,
            expected_parameter.grad,
            rtol=rtol,
            atol=atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _gate_head(gate: Weight, head: int, hidden: torch.Tensor):
    """One head's normalized queries `[T, G, d]` and keys `[T, d]`, as the gate computes them."""

    tokens = hidden.size(0)
    queries = (hidden @ gate.q_proj.weight.T + gate.q_proj.bias).view(
        tokens, gate.nhead, GROUPS, GATE_DIM
    )[:, head]
    keys = (hidden @ gate.k_proj.weight.T).view(tokens, gate.nhead, GATE_DIM)[:, head]
    return gate.q_norm(queries), gate.k_norm(keys)


def _sink_scores(gate: Weight, head: int, queries, keys, logits) -> torch.Tensor:
    sink = torch.einsum("sd,tgd->tsg", gate.k_base[head, 0], queries) / gate.d
    return (1 / (1 + torch.exp(sink - logits.unsqueeze(1)).sum(1))).mean(-1)


def _dense_gate_space_scores(scorer: ImplicitGraphScorer, hidden: torch.Tensor) -> torch.Tensor:
    """The formula with the adjacency materialized: `M = (Y1 Y1^T / T + lambda I) Y2`."""

    mixer = scorer.mixer
    layers, tokens, _ = hidden.shape
    out = torch.empty(layers, scorer.num_heads, tokens, dtype=torch.float64)
    for layer in range(layers):
        gate = scorer.gates[layer]
        x = hidden[layer]
        for head in range(scorer.num_heads):
            graph = layer * scorer.num_heads + head
            y1, y2 = x @ mixer.w1[graph].T, x @ mixer.w2[graph].T
            adjacency = y1 @ y1.T / tokens
            if mixer.self_loop is not None:
                adjacency = adjacency + mixer.self_loop[graph] * torch.eye(tokens, dtype=x.dtype)
            message = adjacency @ y2
            if mixer.normalization == "batchnorm":
                normalized = (message - message.mean(0)) / torch.sqrt(
                    message.var(0, unbiased=False) + 1e-5
                )
                transformed = mixer.gamma[graph] * normalized + mixer.beta[graph]
            else:
                transformed = message
            features = torch.nn.functional.leaky_relu(transformed, mixer.leaky_relu_slope)
            queries, keys = _gate_head(gate, head, x)
            injection = mixer.injection
            if injection.query_proj is not None:
                queries = queries + (features @ injection.query_proj.weight[graph].T).unsqueeze(1)
                keys = keys + features @ injection.key_proj.weight[graph].T
            logits = torch.einsum("tgd,td->tg", queries, keys) / gate.d + gate.b[head, 0]
            if injection.logit_proj is not None:
                logits = logits + features @ injection.logit_proj.weight[graph].T
            out[layer, head] = _sink_scores(gate, head, queries, keys, logits)
    return out.unsqueeze(1)


def _dense_hidden_scores(scorer: ImplicitGraphScorer, hidden: torch.Tensor) -> torch.Tensor:
    """The original residual with self loops: `X' = X + alpha LeakyReLU(Norm((Y1 S + lambda Y2) W))`."""

    mixer = scorer.mixer
    layers, tokens, _ = hidden.shape
    out = torch.empty(layers, scorer.num_heads, tokens, dtype=torch.float64)
    for layer in range(layers):
        gate = scorer.gates[layer]
        x = hidden[layer]
        for head in range(scorer.num_heads):
            graph = layer * scorer.num_heads + head
            y1, y2 = x @ mixer.w1[graph].T, x @ mixer.w2[graph].T
            message = y1 @ (y1.T @ y2 / tokens) + mixer.self_loop[graph] * y2
            pre_activation = message @ mixer.w[graph].T
            if mixer.normalization == "batchnorm":
                normalized = (pre_activation - pre_activation.mean(0)) / torch.sqrt(
                    pre_activation.var(0, unbiased=False) + 1e-5
                )
                transformed = mixer.gamma[graph] * normalized + mixer.beta[graph]
            else:
                transformed = pre_activation
            mixed = x + mixer.alpha[graph] * torch.nn.functional.leaky_relu(
                transformed, mixer.leaky_relu_slope
            )
            queries, keys = _gate_head(gate, head, mixed)
            logits = torch.einsum("tgd,td->tg", queries, keys) / gate.d + gate.b[head, 0]
            out[layer, head] = _sink_scores(gate, head, queries, keys, logits)
    return out.unsqueeze(1)


# --------------------------------------------------------------------------- forward


@pytest.mark.parametrize("normalization", ["none", "batchnorm"])
@pytest.mark.parametrize("target", INJECTION_TARGETS)
def test_gate_space_scores_match_the_dense_self_loop_formula(normalization, target):
    torch.manual_seed(1)
    scorer = _scorer(
        mixer_coupling="gate-space",
        injection_target=target,
        normalization=normalization,
        self_loop_init=0.7,
    )
    _randomize_injection(scorer)
    with torch.no_grad():
        if normalization == "batchnorm":
            scorer.mixer.gamma.normal_(mean=1.0, std=0.3)
            scorer.mixer.beta.normal_(std=0.3)
        scorer.mixer.self_loop.normal_(mean=0.7, std=0.2)
    hidden = _stacked(_example())
    torch.testing.assert_close(
        scorer(hidden, token_microbatch_size=4),
        _dense_gate_space_scores(scorer, hidden),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("normalization", ["none", "batchnorm"])
def test_self_loops_match_the_dense_formula_under_the_hidden_coupling(normalization):
    torch.manual_seed(2)
    scorer = _scorer(normalization=normalization, self_loop_init=0.5)
    with torch.no_grad():
        scorer.mixer.self_loop.normal_(mean=0.5, std=0.2)
        scorer.mixer.alpha.fill_(0.8)
    hidden = _stacked(_example())
    torch.testing.assert_close(
        scorer(hidden, token_microbatch_size=4),
        _dense_hidden_scores(scorer, hidden),
        rtol=1e-10,
        atol=1e-10,
    )


def test_without_the_flag_no_self_loop_exists_and_scores_are_unchanged():
    """The opt-in must leave the original adjacency, and its checkpoints, alone."""

    torch.manual_seed(3)
    plain = _scorer()
    assert plain.mixer.self_loop is None
    looped = copy.deepcopy(plain)
    with torch.no_grad():
        looped.mixer.self_loop = nn.Parameter(torch.zeros(plain.num_graphs, dtype=torch.float64))
    hidden = _stacked(_example())
    torch.testing.assert_close(
        looped(hidden, token_microbatch_size=4), plain(hidden, token_microbatch_size=4)
    )


@pytest.mark.parametrize("architecture", ["implicit", "gps"])
@pytest.mark.parametrize("target", INJECTION_TARGETS)
def test_zero_initialized_injection_scores_exactly_like_the_gate_alone(architecture, target):
    torch.manual_seed(4)
    gates = [_gate(layer, 2) for layer in range(2)]
    options = dict(GPS) if architecture == "gps" else {}
    coupled = ImplicitGraphScorer(
        copy.deepcopy(gates),
        _config(2, 2),
        graph_dim=GRAPH_DIM,
        compute_dtype=torch.float64,
        mixer_coupling="gate-space",
        injection_target=target,
        **options,
    )
    alone = ImplicitGraphScorer(
        copy.deepcopy(gates), _config(2, 2), graph_dim=None, compute_dtype=torch.float64
    )
    hidden = [torch.randn(8, HIDDEN, dtype=torch.float64) for _ in range(2)]
    batch = next(coupled.graph_batches())
    expected = alone.score_subgraph_batch(hidden, batch, (0, 4), 4)
    actual = coupled.score_subgraph_batch(hidden, batch, (0, 4), 4)
    assert torch.equal(actual, expected)
    assert coupled.mixer.alpha is None and coupled.mixer.out_proj is None


@pytest.mark.parametrize("architecture", ["implicit", "gps"])
def test_injection_targets_own_only_their_maps(architecture):
    options = dict(GPS) if architecture == "gps" else {}
    for target, expected in (
        ("qk", {"query_proj", "key_proj"}),
        ("logit", {"logit_proj"}),
        ("qk-logit", {"query_proj", "key_proj", "logit_proj"}),
    ):
        scorer = _scorer(mixer_coupling="gate-space", injection_target=target, **options)
        names = {name.split(".")[0] for name, _ in scorer.mixer.injection.named_parameters()}
        assert names == expected, target
        assert scorer.mixer.injection.target == target
        injection = scorer.mixer.correction_from_prepared(
            scorer.prepare(torch.randn(4, 5, HIDDEN, dtype=torch.float64), tuple(range(4)), token_microbatch_size=5)
        )
        assert isinstance(injection, GateInjection)
        assert (injection.query is None) == ("query_proj" not in expected)
        assert (injection.logit is None) == ("logit_proj" not in expected)


def test_the_adapter_oracle_and_batch_agree_on_an_injection():
    """The single-head statement and the batched path must add the injection alike."""

    torch.manual_seed(5)
    gates = [_gate(layer, 2) for layer in range(2)]
    adapter = _HeadwiseGateAdapter()
    hidden = torch.randn(3, 7, HIDDEN, dtype=torch.float64)
    injection = GateInjection(
        torch.randn(3, 7, GATE_DIM, dtype=torch.float64),
        torch.randn(3, 7, GATE_DIM, dtype=torch.float64),
        torch.randn(3, 7, GROUPS, dtype=torch.float64),
    )
    layer_ids, head_ids = (0, 0, 1), (0, 1, 0)
    batched = adapter.forward_batch(gates, layer_ids, head_ids, hidden, injection)
    for index, (layer, head) in enumerate(zip(layer_ids, head_ids)):
        expected = adapter(
            gates[layer],
            head,
            hidden[index],
            injection=GateInjection(
                injection.query[index], injection.key[index], injection.logit[index]
            ),
        )
        torch.testing.assert_close(batched[index], expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="hidden and delta must match"):
        adapter.forward_batch(gates, layer_ids, head_ids, hidden, injection.select_tokens(torch.arange(3)))


def test_hidden_states_are_materialized_in_the_master_dtype_under_gate_space():
    """With no delta to promote the gate input, the gate-only rule applies."""

    gates = [_gate(layer, 2).to(torch.bfloat16) for layer in range(2)]
    coupled = ImplicitGraphScorer(
        copy.deepcopy(gates), _config(2, 2), graph_dim=GRAPH_DIM, mixer_coupling="gate-space"
    )
    residual = ImplicitGraphScorer(copy.deepcopy(gates), _config(2, 2), graph_dim=GRAPH_DIM)
    alone = ImplicitGraphScorer(copy.deepcopy(gates), _config(2, 2), graph_dim=None)
    assert residual.hidden_dtype == torch.bfloat16
    assert coupled.hidden_dtype == alone.hidden_dtype == torch.float32


# --------------------------------------------------------------------------- training


@pytest.mark.parametrize("normalization", ["none", "batchnorm", "granola"])
@pytest.mark.parametrize("target", ["qk-logit", "logit"])
def test_gate_space_streamed_training_matches_full_autograd(normalization, target):
    torch.manual_seed(6)
    options = dict(GRANOLA) if normalization == "granola" else {"normalization": normalization}
    reference = _scorer(
        mixer_coupling="gate-space",
        injection_target=target,
        self_loop_init=0.5,
        graph_microbatch_size=2,
        **options,
    )
    _randomize_injection(reference)
    streamed = copy.deepcopy(reference)
    example = _example()

    # Both passes must draw the same GraNoLa features; resetting the generator
    # is how the existing gradient tests keep them aligned.
    torch.manual_seed(97)
    expected = _full_loss(reference, example)
    expected.backward()
    trainer = GraphTrainer(
        streamed,
        gate_optimizer=torch.optim.SGD(streamed.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(streamed.mixer.parameters(), lr=0.0),
        token_microbatch_size=4,
        graph_microbatch_size=2,
    )
    torch.manual_seed(97)
    result = trainer.train_context(example, mode="joint")

    torch.testing.assert_close(result["joint_loss"], expected.detach(), rtol=1e-10, atol=1e-10)
    tolerance = dict(rtol=2e-8, atol=2e-9) if normalization == "granola" else dict(rtol=2e-10, atol=2e-10)
    _assert_gradients_close(reference, streamed, **tolerance)
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in streamed.mixer.injection.parameters()
    )


@pytest.mark.parametrize("normalization", ["none", "batchnorm", "granola"])
def test_self_loop_training_matches_full_autograd_under_the_hidden_coupling(normalization):
    torch.manual_seed(7)
    options = dict(GRANOLA) if normalization == "granola" else {"normalization": normalization}
    reference = _scorer(self_loop_init=0.5, graph_microbatch_size=2, **options)
    streamed = copy.deepcopy(reference)
    example = _example()

    torch.manual_seed(98)
    expected = _full_loss(reference, example)
    expected.backward()
    trainer = GraphTrainer(
        streamed,
        gate_optimizer=torch.optim.SGD(streamed.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(streamed.mixer.parameters(), lr=0.0),
        token_microbatch_size=4,
        graph_microbatch_size=2,
    )
    torch.manual_seed(98)
    result = trainer.train_context(example, mode="joint")

    torch.testing.assert_close(result["joint_loss"], expected.detach(), rtol=1e-10, atol=1e-10)
    tolerance = dict(rtol=2e-8, atol=2e-9) if normalization == "granola" else dict(rtol=2e-10, atol=2e-10)
    _assert_gradients_close(reference, streamed, **tolerance)
    assert streamed.mixer.self_loop.grad.abs().sum() > 0


def test_gate_space_scores_and_gradients_are_microbatch_invariant():
    torch.manual_seed(8)
    base = _scorer(mixer_coupling="gate-space", self_loop_init=1.0)
    _randomize_injection(base)
    example = _example()
    hidden = _stacked(example)
    torch.testing.assert_close(
        base(hidden, token_microbatch_size=3, microbatch_size=1),
        base(hidden, token_microbatch_size=9, microbatch_size="auto"),
    )
    trained = []
    for token_size, graph_size in ((3, 1), (9, "auto")):
        scorer = copy.deepcopy(base)
        GraphTrainer(
            scorer,
            mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0.0),
            token_microbatch_size=token_size,
            graph_microbatch_size=graph_size,
        ).train_context(example, mode="joint")
        trained.append(scorer)
    _assert_gradients_close(trained[0].mixer, trained[1].mixer, rtol=1e-9, atol=1e-9)


def test_gps_gate_space_trains_with_autograd_and_still_requires_subgraphs():
    torch.manual_seed(9)
    scorer = _scorer(mixer_coupling="gate-space", **GPS)
    example = _example()
    with pytest.raises(ValueError, match="fixed-size subgraphs"):
        GraphTrainer(scorer, token_microbatch_size=9)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0.0),
        token_microbatch_size=9,
        subgraph_size=3,
    )
    result = trainer.train_context(example, mode="joint")
    assert torch.isfinite(result["joint_loss"])
    assert all(
        parameter.grad is not None for parameter in scorer.mixer.injection.parameters()
    )
    assert scorer.mixer.alpha is None and scorer.mixer.out_proj is None


def test_gate_space_works_through_the_answer_training_scoring_and_replay_path():
    torch.manual_seed(10)
    for options, subgraph_size in (({}, None), ({}, 3), (dict(GPS), 3)):
        scorer = _scorer(mixer_coupling="gate-space", self_loop_init=None if options else 0.5, **options)
        hidden = [torch.randn(9, HIDDEN, dtype=torch.float64) for _ in range(2)]
        scores = score_context_subgraphs(
            scorer, hidden, subgraph_size=subgraph_size, token_microbatch_size=9
        )
        assert tuple(scores.shape) == (2, 1, 2, 9)
        replay_score_gradients(
            scorer,
            hidden,
            torch.randn_like(scores),
            torch.zeros(2, 1, 2, 2, dtype=torch.long),
            subgraph_size=subgraph_size,
            token_microbatch_size=9,
        )
        missing = [name for name, value in scorer.mixer.named_parameters() if value.grad is None]
        assert missing == [], (options, subgraph_size)


@pytest.mark.parametrize("self_loop_init", [None, 1.0])
def test_decay_membership_covers_each_gate_space_parameter_once(self_loop_init):
    scorer = _scorer(mixer_coupling="gate-space", self_loop_init=self_loop_init)
    _, optimizer = build_adamw_optimizers(scorer, weight_decay=0.2)
    names_by_id = {id(parameter): name for name, parameter in scorer.mixer.named_parameters()}
    decay = {names_by_id[id(p)] for p in optimizer.param_groups[0]["params"]}
    no_decay = {names_by_id[id(p)] for p in optimizer.param_groups[1]["params"]}
    assert decay == {
        "in_proj.weight",
        "injection.query_proj.weight",
        "injection.key_proj.weight",
        "injection.logit_proj.weight",
    }
    assert no_decay == {"gamma", "beta"} | ({"self_loop"} if self_loop_init is not None else set())
    assert decay | no_decay == set(names_by_id.values())


# --------------------------------------------------------------------------- checkpoints


def _checkpoint_config(scorer: ImplicitGraphScorer, **overrides):
    config = {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": GATE_DIM,
        "gate_sink": SINK,
        "hidden_dim": HIDDEN,
        "num_layers": scorer.num_layers,
        "num_kv_heads": scorer.num_heads,
        "query_groups": GROUPS,
        "graph_dim": GRAPH_DIM,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 9,
        "gram_normalization": "token-count",
        "normalization": scorer.mixer.normalization,
        "normalization_sharing": "graph",
        "granola_gnn_depth": 1,
        "granola_mlp_depth": 1,
        "granola_rnf_dim": 1,
        "granola_adaptivity": "graph",
        "normalization_seed": 0,
        "leaky_relu_slope": 0.01,
        "activation_order": ACTIVATION_ORDER,
        "mixer_architecture": "implicit",
        **scorer.mixer.coupling_config(),
    }
    if scorer.mixer.coupling == "hidden":
        config["alpha_init"] = 0.1
    config.update(overrides)
    return config


def _save(tmp_path, scorer, config, name="last"):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return save_checkpoint(
        directory,
        "last",
        scorer=scorer,
        config=config,
        model_id="unit",
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor={"epoch": 0},
        wandb_run_id=None,
    )


@pytest.mark.parametrize("target", INJECTION_TARGETS)
def test_gate_space_checkpoint_round_trips_and_the_evaluator_reproduces_scores(tmp_path, target):
    from graph.evaluation import load_evaluation_checkpoint, reconstruct_graph_scorer

    torch.manual_seed(11)
    scorer = _scorer(mixer_coupling="gate-space", injection_target=target, self_loop_init=0.3)
    _randomize_injection(scorer)
    hidden = _stacked(_example())
    expected = scorer(hidden, token_microbatch_size=9)
    path = _save(tmp_path, scorer, _checkpoint_config(scorer))

    restored = _scorer(mixer_coupling="gate-space", injection_target=target, self_loop_init=0.3)
    load_checkpoint(path, scorer=restored, restore_rng=False)
    torch.testing.assert_close(restored(hidden, token_microbatch_size=9), expected)

    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.mixer_coupling == "gate-space"
    assert "alpha_init" not in checkpoint.config
    rebuilt = reconstruct_graph_scorer(
        checkpoint, SimpleNamespace(config=_config(2, 2), device="cpu", gates=None)
    )
    assert rebuilt.mixer.injection.target == target
    assert rebuilt.mixer.self_loop is not None
    torch.testing.assert_close(rebuilt(hidden, token_microbatch_size=9), expected)


def test_a_checkpoint_without_a_coupling_key_loads_as_hidden(tmp_path):
    from graph.evaluation import load_evaluation_checkpoint, reconstruct_graph_scorer

    torch.manual_seed(12)
    scorer = _scorer()
    config = _checkpoint_config(scorer)
    del config["mixer_coupling"]
    path = _save(tmp_path, scorer, config)
    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.mixer_coupling == "hidden"
    assert checkpoint.config["mixer_coupling"] == "hidden"
    restored = _scorer()
    load_checkpoint(path, scorer=restored, restore_rng=False)
    rebuilt = reconstruct_graph_scorer(
        checkpoint, SimpleNamespace(config=_config(2, 2), device="cpu", gates=None)
    )
    assert rebuilt.mixer.coupling == "hidden" and rebuilt.mixer.self_loop is None


def test_coupling_and_self_loop_mismatches_are_refused_by_name(tmp_path):
    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(13)
    hidden_path = _save(tmp_path, _scorer(), _checkpoint_config(_scorer()), "hidden")
    with pytest.raises(ValueError, match="coupling configuration conflicts.*mixer_coupling"):
        load_checkpoint(hidden_path, scorer=_scorer(mixer_coupling="gate-space"), restore_rng=False)
    with pytest.raises(ValueError, match="coupling configuration conflicts.*self_loop_init"):
        load_checkpoint(hidden_path, scorer=_scorer(self_loop_init=1.0), restore_rng=False)

    coupled = _scorer(mixer_coupling="gate-space")
    coupled_path = _save(tmp_path, coupled, _checkpoint_config(coupled), "coupled")
    with pytest.raises(ValueError, match="coupling configuration conflicts.*mixer_coupling"):
        load_checkpoint(coupled_path, scorer=_scorer(), restore_rng=False)

    config = _checkpoint_config(coupled)
    del config["injection_target"]
    with pytest.raises(ValueError, match="missing: injection_target"):
        load_evaluation_checkpoint(_save(tmp_path, coupled, config, "missing-target"))
    config = _checkpoint_config(coupled)
    config["alpha_init"] = 0.1
    stale = _save(tmp_path, coupled, config, "stale-alpha")
    # A setting nothing applies is tolerated on read; the writer never emits it.
    assert load_evaluation_checkpoint(stale).mixer_coupling == "gate-space"


# --------------------------------------------------------------------------- CLI contract


def _train_args(*extra):
    import train_graph

    return train_graph.build_parser().parse_args(["--model", "unit", *extra])


def _answer_args(*extra):
    import train_graph_answer

    return train_graph_answer.build_parser().parse_args(
        ["--model", "Qwen/unit", "--validation-retention-ratio", "0.2", *extra]
    )


def _recorded(*flags):
    import train_graph

    options = train_graph.resolve_options(_train_args("--graph-dim", "4", *flags))
    config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=_scorer(layers=1, heads=1), options=options, query_groups=GROUPS
    )
    return {
        key: config.get(key, "absent")
        for key in ("mixer_coupling", "injection_target", "self_loop_init", "alpha_init")
    }


def test_the_checkpoint_config_records_coupling_settings_only_when_applied():
    assert _recorded() == {
        "mixer_coupling": "hidden",
        "injection_target": "absent",
        "self_loop_init": "absent",
        "alpha_init": 0.1,
    }
    assert _recorded("--mixer-coupling", "gate-space") == {
        "mixer_coupling": "gate-space",
        "injection_target": "qk-logit",
        "self_loop_init": "absent",
        "alpha_init": "absent",
    }
    assert _recorded(
        "--mixer-coupling", "gate-space", "--injection-target", "logit", "--self-loop-init", "1"
    ) == {
        "mixer_coupling": "gate-space",
        "injection_target": "logit",
        "self_loop_init": 1.0,
        "alpha_init": "absent",
    }
    assert _recorded("--self-loop-init", "0") == {
        "mixer_coupling": "hidden",
        "injection_target": "absent",
        "self_loop_init": 0.0,
        "alpha_init": 0.1,
    }


def test_both_scripts_refuse_settings_nothing_applies():
    import train_graph
    import train_graph_answer

    cases = (
        (("--injection-target", "qk"), "requires --mixer-coupling gate-space"),
        (
            ("--mixer-coupling", "gate-space", "--alpha-init", "0.5"),
            "gate-space coupling has no residual weight",
        ),
        (
            (
                "--mixer-architecture", "gps", "--subgraph-size", "4",
                "--token-microbatch-size", "4", "--gps-attention-heads", "2",
                "--self-loop-init", "1",
            ),
            "applies to the implicit mixer",
        ),
    )
    for flags, message in cases:
        with pytest.raises(ValueError, match=message):
            train_graph.resolve_options(_train_args(*flags))
        with pytest.raises(ValueError, match=message):
            train_graph_answer.resolve_options(_answer_args(*flags))
    with pytest.raises(ValueError, match="requires the graph mixer"):
        train_graph_answer.resolve_options(
            _answer_args("--no-graph-mixer", "--mixer-coupling", "gate-space")
        )


def test_gate_space_resolves_for_both_architectures_and_every_normalization():
    import train_graph

    for architecture in ("implicit", "gps"):
        for normalization in ("none", "batchnorm", "granola"):
            if architecture == "gps" and normalization != "none":
                continue
            flags = ["--mixer-coupling", "gate-space", "--normalization", normalization]
            if architecture == "gps":
                flags += [
                    "--mixer-architecture", "gps", "--subgraph-size", "4",
                    "--token-microbatch-size", "4", "--gps-attention-heads", "2",
                ]
            options = train_graph.resolve_options(_train_args("--graph-dim", "4", *flags))
            assert options.mixer_coupling == "gate-space"
            assert options.injection_target == "qk-logit"
            assert options.self_loop_init is None


def test_resume_cannot_switch_the_coupling_or_the_target():
    import train_graph
    import train_graph_answer

    saved = {
        **_checkpoint_config(_scorer(mixer_coupling="gate-space")),
        "training_mode": "joint",
        "gate_lr": 1e-4,
        "mixer_lr": 1e-3,
    }
    with pytest.raises(ValueError, match="mixer_coupling"):
        train_graph.resolve_options(
            _train_args("--mixer-coupling", "hidden"),
            {"config": saved, "model_id": "unit", "prefill_chunk": 4},
        )
    with pytest.raises(ValueError, match="injection_target"):
        train_graph.resolve_options(
            _train_args("--injection-target", "logit"),
            {"config": saved, "model_id": "unit", "prefill_chunk": 4},
        )
    answer_saved = {
        **saved,
        "model_id": "Qwen/unit",
        "data": "agentic",
        "validation_retention_ratio": 0.2,
    }
    with pytest.raises(ValueError, match="mixer_coupling|conflict"):
        train_graph_answer.resolve_options(
            _answer_args("--graph-checkpoint", "source.pt", "--mixer-coupling", "hidden"),
            {"config": answer_saved, "model_id": "Qwen/unit", "prefill_chunk": 4},
        )
    options = train_graph_answer.resolve_options(
        _answer_args("--graph-checkpoint", "source.pt"),
        {"config": answer_saved, "model_id": "Qwen/unit", "prefill_chunk": 4},
    )
    assert options.mixer_coupling == "gate-space"
    assert options.injection_target == "qk-logit"


def test_stage_one_logs_the_self_loop_mean_instead_of_alpha_under_gate_space(monkeypatch):
    import train_graph

    parameter = nn.Parameter(torch.zeros(()))
    trainer = SimpleNamespace(
        scorer=SimpleNamespace(
            device=torch.device("cpu"),
            mixer=SimpleNamespace(self_loop=torch.tensor([1.0, 3.0])),
        ),
        gate_optimizer=torch.optim.SGD([parameter], lr=0.01),
        mixer_optimizer=torch.optim.SGD([parameter], lr=0.02),
        timing=None,
        train_context=lambda *_args, **_kwargs: {
            "joint_loss": 0.5,
            "gate_loss": None,
            "graph_loss": None,
            "gradient_norm": 3.0,
            "gate_gradient_norm": 1.0,
            "mixer_gradient_norm": 2.0,
        },
    )
    monkeypatch.setattr(
        train_graph, "PhaseTiming", lambda *_: SimpleNamespace(resolve=lambda: {})
    )
    _, metrics = train_graph.run_and_log_context(
        trainer,
        _example(1, 1),
        mode="joint",
        validation=False,
        run=SimpleNamespace(log=lambda metrics, *, step: None),
        step=1,
    )
    assert metrics["train/mean_self_loop"] == 2.0
    assert "train/mean_alpha" not in metrics


# --------------------------------------------------------- gate-input dtype


class _WideNorm(nn.Module):
    """RMSNorm whose weight is wider than its input, like the released gate's.

    The released FastKVzip gate stores its projections in bfloat16 and its
    query/key norms in float32, and the scorer keeps every gate parameter at
    the master dtype. Multiplying by that weight returns the wider dtype, so a
    caller that hands the gate a narrower hidden state gets a wider result
    back from the norm than it put in.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = values.to(torch.float32) * torch.rsqrt(variance + 1e-6)
        return self.weight * normalized.to(values.dtype)


def _low_precision_scorer(**options):
    """A scorer whose compute dtype is narrower than its gates' master dtype."""

    gates = [_gate(layer, 2).to(torch.bfloat16) for layer in range(2)]
    for gate in gates:
        gate.q_norm = _WideNorm(GATE_DIM)
        gate.k_norm = _WideNorm(GATE_DIM)
    return ImplicitGraphScorer(
        gates,
        _config(2, 2),
        graph_dim=GRAPH_DIM,
        graph_microbatch_size="auto",
        **options,
    )


def test_the_adapter_agrees_with_its_oracle_when_the_norm_widens_the_dtype():
    """A widening norm must not make the batched path disagree with one head."""

    torch.manual_seed(30)
    gates = [_gate(layer, 2).to(torch.bfloat16) for layer in range(2)]
    for gate in gates:
        gate.q_norm = _WideNorm(GATE_DIM)
        gate.k_norm = _WideNorm(GATE_DIM)
    adapter = _HeadwiseGateAdapter()
    hidden = torch.randn(3, 7, HIDDEN, dtype=torch.bfloat16)
    layer_ids, head_ids = (0, 0, 1), (0, 1, 0)

    batched = adapter.forward_batch(gates, layer_ids, head_ids, hidden)

    for index, (layer, head) in enumerate(zip(layer_ids, head_ids)):
        expected = adapter(gates[layer], head, hidden[index])
        torch.testing.assert_close(
            batched[index].to(expected.dtype), expected, rtol=2e-2, atol=2e-2
        )


def test_both_ways_into_the_gate_score_a_low_precision_run_identically():
    """The trainer's own call into the gate must match `score_prepared`'s.

    `score_prepared` materializes the hidden states in the scorer's hidden
    dtype, which is the gates' master dtype. The trainer reaches the same
    adapter directly and has to do the same, or the gate's projections run at
    the narrower compute dtype down one path and the master dtype down the
    other, and the two disagree on the scores they produce.

    The hidden coupling hid this: its delta is accumulated at the master dtype,
    so adding it promoted the gate input whichever way it was reached. The
    gate-space coupling adds nothing to the gate input, so the gap is real.
    """

    torch.manual_seed(31)
    scorer = _low_precision_scorer(mixer_coupling="gate-space")
    assert scorer.compute_dtype == torch.bfloat16
    assert scorer.hidden_dtype == torch.float32
    _randomize_injection(scorer, std=0.05)

    example = _example(tokens=6)
    trainer = GraphTrainer(scorer, token_microbatch_size=6, graph_microbatch_size=2)
    batch = next(scorer.graph_batches(microbatch_size=2))
    positions = torch.arange(6)
    hidden = trainer._hidden(example, batch.layer_ids, positions)
    assert hidden.dtype == scorer.compute_dtype

    prepared = scorer.prepare(hidden, batch.graph_ids, token_microbatch_size=6)
    correction = scorer.mixer.correction_from_prepared(prepared)
    expected, _ = scorer.score_prepared(
        hidden, prepared, layer_ids=batch.layer_ids, head_ids=batch.head_ids
    )
    actual = trainer._score_from_correction(hidden, correction, batch)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("architecture", ["implicit", "gps"])
def test_low_precision_gate_space_training_runs(architecture):
    """The end-to-end shape of the pilot that failed: a bfloat16 gate."""

    torch.manual_seed(31)
    options = dict(GPS) if architecture == "gps" else {}
    scorer = _low_precision_scorer(mixer_coupling="gate-space", **options)
    _randomize_injection(scorer, std=0.05)

    example = _example(tokens=6)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0.0),
        token_microbatch_size=3,
        graph_microbatch_size=2,
        **({"subgraph_size": 3} if architecture == "gps" else {}),
    )
    result = trainer.train_context(example, mode="joint")

    assert torch.isfinite(result["joint_loss"])
    assert all(
        parameter.grad is not None for parameter in scorer.mixer.injection.parameters()
    )


def test_a_low_precision_hidden_coupling_run_still_trains():
    """The coupling this fix was found under is not the only one that uses it."""

    torch.manual_seed(32)
    scorer = _low_precision_scorer(self_loop_init=1.0)
    example = _example(tokens=6)
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0.0),
        token_microbatch_size=3,
        graph_microbatch_size=2,
    )
    result = trainer.train_context(example, mode="joint")
    assert torch.isfinite(result["joint_loss"])
    assert scorer.mixer.self_loop.grad is not None


# ------------------------------------------------- how the injection starts


def _mixer_body_gradient(scorer, example, **trainer_options):
    """Largest gradient reaching the mixer's own projection, not its maps."""

    trainer = GraphTrainer(
        scorer,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.0),
        mixer_optimizer=torch.optim.SGD(scorer.mixer.parameters(), lr=0.0),
        token_microbatch_size=4,
        graph_microbatch_size=2,
        **trainer_options,
    )
    trainer.train_context(example, mode="joint")
    body = scorer.mixer.in_proj.weight.grad
    maps = scorer.mixer.injection.query_proj.weight.grad
    return float(body.abs().max()), float(maps.abs().max())


def test_zero_initialized_maps_leave_the_mixer_body_without_gradient():
    """Why a nonzero start exists at all.

    The mixer's gradient arrives through the injection maps. With the maps at
    zero the body gets exactly nothing and cannot train until they grow, the
    same shape of problem an alpha of zero causes under the hidden coupling.
    The maps themselves still train, so this is a slow start rather than a
    permanent freeze, but a fixed-length run spends much of its budget on it.
    """

    torch.manual_seed(40)
    scorer = _scorer(mixer_coupling="gate-space", graph_microbatch_size=2)
    assert scorer.mixer.injection.init_std == 0.0
    body, maps = _mixer_body_gradient(scorer, _example())

    assert body == 0.0
    assert maps > 0.0


@pytest.mark.parametrize("architecture", ["implicit", "gps"])
def test_a_nonzero_start_gives_the_mixer_body_gradient_immediately(architecture):
    torch.manual_seed(41)
    options = dict(GPS) if architecture == "gps" else {}
    scorer = _scorer(
        mixer_coupling="gate-space",
        injection_init=0.02,
        graph_microbatch_size=2,
        **options,
    )
    extra = {"subgraph_size": 2} if architecture == "gps" else {}
    body, maps = _mixer_body_gradient(scorer, _example(tokens=6), **extra)

    assert body > 0.0
    assert maps > 0.0


def test_the_injection_init_sets_the_scale_the_maps_start_at():
    torch.manual_seed(42)
    for std in (0.01, 0.1):
        scorer = _scorer(mixer_coupling="gate-space", injection_init=std)
        weights = torch.cat(
            [p.flatten() for p in scorer.mixer.injection.parameters()]
        )
        # A few hundred entries, so allow the sample deviation some room.
        assert weights.numel() > 100
        assert abs(float(weights.std()) - std) < 0.3 * std
        assert scorer.mixer.injection.init_std == std


def test_a_nonzero_start_no_longer_scores_exactly_like_the_gate_alone():
    """The trade the flag makes, stated as a test.

    Zero keeps the exact gate-only start; nonzero gives that up to wake the
    mixer. A caller choosing the trade should see both halves of it.
    """

    torch.manual_seed(43)
    gates = [_gate(layer, 2) for layer in range(2)]
    alone = ImplicitGraphScorer(
        [copy.deepcopy(g) for g in gates], _config(2, 2), graph_dim=None,
        compute_dtype=torch.float64,
    )
    hidden = [torch.randn(8, HIDDEN, dtype=torch.float64) for _ in range(2)]
    batch = next(alone.graph_batches())
    expected = alone.score_subgraph_batch(hidden, batch, (0, 4), 4)

    for std, identical in ((0.0, True), (0.05, False)):
        scorer = ImplicitGraphScorer(
            [copy.deepcopy(g) for g in gates],
            _config(2, 2),
            graph_dim=GRAPH_DIM,
            compute_dtype=torch.float64,
            mixer_coupling="gate-space",
            injection_init=std,
        )
        actual = scorer.score_subgraph_batch(hidden, batch, (0, 4), 4)
        assert torch.equal(actual, expected) is identical, std


def test_the_injection_init_is_recorded_and_refused_where_it_does_not_apply():
    import train_graph
    import train_graph_answer

    assert _recorded("--mixer-coupling", "gate-space", "--injection-init", "0.02") == {
        "mixer_coupling": "gate-space",
        "injection_target": "qk-logit",
        "self_loop_init": "absent",
        "alpha_init": "absent",
    }
    options = train_graph.resolve_options(
        _train_args("--graph-dim", "4", "--mixer-coupling", "gate-space",
                    "--injection-init", "0.02")
    )
    assert options.injection_init == 0.02
    config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=_scorer(layers=1, heads=1), options=options,
        query_groups=GROUPS,
    )
    assert config["injection_init"] == 0.02

    for flags in (("--injection-init", "0.02"),):
        with pytest.raises(ValueError, match="requires --mixer-coupling gate-space"):
            train_graph.resolve_options(_train_args("--graph-dim", "4", *flags))
        with pytest.raises(ValueError, match="requires --mixer-coupling gate-space"):
            train_graph_answer.resolve_options(_answer_args(*flags))
    with pytest.raises(ValueError, match="finite and non-negative"):
        train_graph.resolve_options(
            _train_args("--graph-dim", "4", "--mixer-coupling", "gate-space",
                        "--injection-init", "-1")
        )


# ---------------------------------------------- what the ranking metric sees


def test_topk_overlap_reads_the_order_and_ignores_the_scale():
    """The property that makes it worth logging next to BCE.

    Eviction depends only on the order of the scores, so a monotone rescaling
    must not move this metric even though it moves BCE a lot, and a swap
    across the retention threshold must move it even though BCE barely
    notices.
    """

    from graph import topk_overlap

    torch.manual_seed(60)
    scores = torch.rand(4, 100, dtype=torch.float64)
    perfect = topk_overlap(scores, scores, ratios=(0.1, 0.3))
    assert perfect == {0.1: 1.0, 0.3: 1.0}

    # Monotone rescaling: every ranking identical, BCE very different.
    squashed = scores.pow(3) * 0.5
    assert topk_overlap(squashed, scores, ratios=(0.1, 0.3)) == perfect
    bce = torch.nn.functional.binary_cross_entropy
    assert bce(squashed, scores) - bce(scores, scores) > 0.05

    # A swap across the threshold: ranking changes, BCE hardly moves.
    swapped = scores.clone()
    order = scores[0].argsort(descending=True)
    top, just_below = order[9], order[10]
    swapped[0, top], swapped[0, just_below] = scores[0, just_below], scores[0, top]
    moved = topk_overlap(swapped, scores, ratios=(0.1,))
    assert moved[0.1] < perfect[0.1]
    assert abs(float(bce(swapped, scores) - bce(scores, scores))) < 0.01


def test_validation_reports_the_overlap_for_every_evaluated_ratio():
    from graph import TOPK_OVERLAP_RATIOS

    torch.manual_seed(61)
    scorer = _scorer(mixer_coupling="gate-space", injection_init=0.05)
    trainer = GraphTrainer(scorer, token_microbatch_size=4, graph_microbatch_size=2)
    result = trainer.evaluate_context(_example(tokens=9))

    assert set(result.topk_overlap) == set(TOPK_OVERLAP_RATIOS)
    assert all(0.0 <= v <= 1.0 for v in result.topk_overlap.values())


def test_the_overlap_is_scored_over_the_whole_context_not_per_chunk():
    """Ranking is a whole-context property, so the token split must not move it."""

    torch.manual_seed(62)
    base = _scorer(mixer_coupling="gate-space", injection_init=0.05)
    example = _example(tokens=12)
    whole = GraphTrainer(base, token_microbatch_size=12, graph_microbatch_size=2)
    split = GraphTrainer(base, token_microbatch_size=3, graph_microbatch_size=2)

    assert whole.evaluate_context(example).topk_overlap == pytest.approx(
        split.evaluate_context(example).topk_overlap
    )


def test_a_wide_logit_gap_still_yields_a_finite_gradient():
    """A saturated gate must not poison training with a NaN.

    Run 21477027 died here. The keep probability used to be written as
    1 / (1 + sum exp(gap)); once the gap passed about 88 the exponential
    overflowed, the score underflowed to zero, and the backward pass
    multiplied a zero local derivative by an infinite one and produced NaN.
    The optimizer wrote that NaN into every parameter and the next context
    tripped the loss function's domain check.
    """

    from graph.model import _keep_probability

    for gap in (10.0, 50.0, 90.0, 400.0):
        base = torch.full((1, 4, 1), gap, dtype=torch.float32, requires_grad=True)
        logits = torch.zeros(1, 1, 1, dtype=torch.float32)

        score = _keep_probability(base, logits, dim=1)
        score.sum().backward()

        assert torch.isfinite(score).all(), f"score not finite at gap {gap}"
        assert torch.isfinite(base.grad).all(), f"gradient not finite at gap {gap}"
        assert 0.0 <= float(score) <= 1.0, f"score outside [0,1] at gap {gap}"


def test_the_stable_keep_probability_matches_the_direct_formula():
    """The rewrite must be the same function, not an approximation."""

    from graph.model import _keep_probability

    torch.manual_seed(63)
    base = torch.randn(3, 5, 2, dtype=torch.float64)
    logits = torch.randn(3, 1, 2, dtype=torch.float64)

    direct = 1 / (1 + torch.exp(base - logits).sum(dim=1))
    assert _keep_probability(base, logits, dim=1) == pytest.approx(direct)


def test_a_gate_only_run_can_train_and_validate_without_a_mixer():
    """The control every mixer row is measured against must actually run.

    Evaluation used to reach for the mixer unconditionally and fail on a
    gate-only scorer, so there was no baseline to compare a mixer against.
    """

    from graph import TOPK_OVERLAP_RATIOS

    torch.manual_seed(64)
    scorer = _scorer(graph_dim=None)
    assert scorer.mixer is None

    trainer = GraphTrainer(
        scorer,
        token_microbatch_size=4,
        graph_microbatch_size=2,
        gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.01),
    )
    example = _example(tokens=9)

    trained = trainer.train_context(example, mode="gate")
    assert trained["gate_loss"] is not None

    validation = trainer.evaluate_context(example)
    assert math.isfinite(float(validation.loss))
    assert set(validation.topk_overlap) == set(TOPK_OVERLAP_RATIOS)
    assert all(0.0 <= v <= 1.0 for v in validation.topk_overlap.values())


def test_the_injection_share_tracks_how_hard_the_mixer_pushes_the_gate():
    """The share must rise with the injection scale, and vanish without one.

    Weight norms proved unable to answer this: a finished run held its
    starting norm to within one percent while saying nothing about whether
    the term it produced still reached the gate. This measures the term.
    """

    def share_at(init):
        torch.manual_seed(70)
        scorer = _scorer(mixer_coupling="gate-space", injection_init=init)
        trainer = GraphTrainer(
            scorer, token_microbatch_size=4, graph_microbatch_size=2
        )
        trainer.evaluate_context(_example(tokens=9))
        return scorer._gate_adapter.consume_injection_share()

    quiet = share_at(0.01)
    loud = share_at(0.4)

    assert set(quiet) == {"query", "key", "logit"}
    for target in quiet:
        assert quiet[target] > 0.0
        assert loud[target] > quiet[target] * 5, (
            f"{target}: {loud[target]:.4f} should dwarf {quiet[target]:.4f}"
        )


def test_a_zero_injection_contributes_nothing_to_the_gate():
    torch.manual_seed(71)
    scorer = _scorer(mixer_coupling="gate-space", injection_init=0.0)
    trainer = GraphTrainer(scorer, token_microbatch_size=4, graph_microbatch_size=2)
    trainer.evaluate_context(_example(tokens=9))

    assert all(v == 0.0 for v in scorer._gate_adapter.consume_injection_share().values())


def test_reading_the_injection_share_resets_it():
    """It accumulates across chunks, so a stale tally would corrupt the next."""

    torch.manual_seed(72)
    scorer = _scorer(mixer_coupling="gate-space", injection_init=0.2)
    trainer = GraphTrainer(scorer, token_microbatch_size=4, graph_microbatch_size=2)
    trainer.evaluate_context(_example(tokens=9))

    assert scorer._gate_adapter.consume_injection_share()
    assert scorer._gate_adapter.consume_injection_share() == {}


def test_the_injection_share_does_not_depend_on_the_token_split():
    """It is a property of the context, not of how the context was chunked."""

    torch.manual_seed(73)
    scorer = _scorer(mixer_coupling="gate-space", injection_init=0.2)
    example = _example(tokens=12)

    def share_with(token_microbatch):
        GraphTrainer(
            scorer,
            token_microbatch_size=token_microbatch,
            graph_microbatch_size=2,
        ).evaluate_context(example)
        return scorer._gate_adapter.consume_injection_share()

    # Summing squares in a different chunk order moves the last few bits.
    assert share_with(12) == pytest.approx(share_with(3), rel=1e-6)


def test_drift_sees_a_rotation_that_leaves_the_norm_alone():
    """The reason drift replaced scale: a weight can move without growing."""

    from train_graph import _relative_drift

    weight = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    reference: dict = {}

    assert _relative_drift(weight, reference) == 0.0
    with torch.no_grad():
        # Same norm, opposite direction: scale reports no change at all.
        weight.copy_(torch.tensor([-3.0, -4.0]))
    assert _relative_drift(weight, reference) == pytest.approx(2.0)


def test_a_gate_only_scorer_trains_under_any_mode():
    """The control is asked for the same way as any other run.

    The CLI only offers two-phase and joint, and both used to reach for a
    mixer phase that a gate-only scorer cannot run. There was no way to ask
    for the baseline at all.
    """

    for mode in ("joint", "two_phase", "gate"):
        torch.manual_seed(74)
        scorer = _scorer(graph_dim=None)
        trainer = GraphTrainer(
            scorer,
            token_microbatch_size=4,
            graph_microbatch_size=2,
            gate_optimizer=torch.optim.SGD(scorer.gates.parameters(), lr=0.01),
        )
        result = trainer.train_context(_example(tokens=9), mode=mode)
        assert result["gate_loss"] is not None, mode
        assert math.isfinite(float(result["gate_loss"])), mode
