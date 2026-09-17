import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graph import (
    ImplicitGraphMixer,
    GPS_ACTIVATION_ORDER,
    GPSGraphMixer,
    GraphTrainer,
    ImplicitGraphScorer,
    TeacherExample,
    build_adamw_optimizers,
    mixer_activation_order,
)
import torch.nn.functional as F

from graph.model import (
    sinusoidal_positions,
    _GPSBlock,
    _PerGraphImplicitBranch,
    _PerGraphPerformerAttention,
    PerGraphLayerNorm,
    orthogonal_random_features,
)


class _ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        return value * self.weight


class Gate(nn.Module):
    def __init__(self, hidden=4, heads=1):
        super().__init__()
        self.nhead, self.ngroup, self.output_dim, self.sink = heads, 1, 1, 1
        self.d = 1.0
        self.q_proj = nn.Linear(hidden, heads, bias=True)
        self.k_proj = nn.Linear(hidden, heads, bias=False)
        self.q_norm = _ScaleNorm(1)
        self.k_norm = _ScaleNorm(1)
        self.k_base = nn.Parameter(torch.ones(heads, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(heads, 1, 1))


def _config(layers=1, heads=1, hidden=4):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads,
        hidden_size=hidden,
    )


def _scorer(layers=1, heads=1, hidden=4, architecture="gps", **kwargs):
    options = dict(
        graph_dim=4,
        mixer_architecture=architecture,
        gps_depth=2,
        gps_attention_heads=2,
        gps_random_features=8,
        graph_microbatch_size="auto",
        compute_dtype=torch.float64,
    )
    options.update(kwargs)
    return ImplicitGraphScorer(
        [Gate(hidden=hidden, heads=heads).double() for _ in range(layers)],
        _config(layers, heads, hidden),
        **options,
    )


def _example(layers=1, heads=1, tokens=8, hidden=4):
    return TeacherExample(
        dataset_name="unit",
        dataset_index=0,
        token_ids=torch.arange(tokens).view(1, -1),
        hidden_by_layer=[
            torch.randn(tokens, hidden, dtype=torch.float64) for _ in range(layers)
        ],
        teacher_scores=torch.rand(layers, 1, heads, tokens, dtype=torch.float64),
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        sequence_length=tokens,
    )


def test_random_features_estimate_the_softmax_kernel_without_bias():
    """The orthogonal draw must stay Haar-uniform, or the kernel is biased."""

    torch.manual_seed(0)
    dimension = 8
    left = torch.randn(dimension, dtype=torch.float64) * 0.3
    right = torch.randn(dimension, dtype=torch.float64) * 0.3
    exact = torch.exp(left @ right)

    def estimate(features):
        omega = orthogonal_random_features(
            features, dimension, device=None, dtype=torch.float64
        )
        return (
            torch.exp(omega @ left - left @ left / 2)
            * torch.exp(omega @ right - right @ right / 2)
        ).mean()

    coarse = (estimate(256) - exact).abs()
    fine = (estimate(65536) - exact).abs()
    assert fine < coarse
    assert fine / exact < 0.02


def test_performer_branch_converges_to_exact_softmax_attention():
    graphs, tokens, width, heads = 2, 40, 16, 2
    errors = []
    for features in (64, 1024, 16384):
        torch.manual_seed(7)
        attention = _PerGraphPerformerAttention(graphs, width, heads, features).double()
        hidden = torch.randn(graphs, tokens, width, dtype=torch.float64)
        approximate = attention(hidden, (0, 1))

        packed = attention.qkv_proj(hidden, (0, 1))
        query, key, value = (
            part.view(graphs, tokens, heads, width // heads)
            for part in packed.split(width, dim=-1)
        )
        logits = torch.einsum("gthd,gshd->ghts", query, key) / math.sqrt(width // heads)
        exact = attention.out_proj(
            torch.einsum("ghts,gshd->gthd", logits.softmax(-1), value).reshape(
                graphs, tokens, -1
            ),
            (0, 1),
        )
        errors.append(float(((approximate - exact).norm() / exact.norm()).detach()))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.06


def test_performer_branch_stays_finite_in_bfloat16():
    """Production runs in bf16, where an unstabilized exponential underflows."""

    torch.manual_seed(31)
    graphs, tokens, width, heads = 2, 256, 32, 4
    attention = _PerGraphPerformerAttention(graphs, width, heads, 64)
    hidden = torch.randn(graphs, tokens, width) * 3.0
    exact = attention.double()(hidden.double(), (0, 1))
    low = attention.to(torch.bfloat16)(hidden.to(torch.bfloat16), (0, 1))

    assert torch.isfinite(low).all()
    assert float(((low.double() - exact).norm() / exact.norm()).detach()) < 0.05


def test_gps_scores_each_subgraph_independently():
    """Tokens of one subgraph must never influence another subgraph's scores."""

    torch.manual_seed(1)
    scorer = _scorer(layers=2, heads=2)
    hidden = [torch.randn(12, 4, dtype=torch.float64) for _ in range(2)]
    batch = next(scorer.graph_batches())
    before = scorer.score_subgraph_batch(hidden, batch, (0, 4, 8), 4)

    changed = [layer.clone() for layer in hidden]
    for layer in changed:
        layer[4:8] = torch.randn(4, 4, dtype=torch.float64)
    after = scorer.score_subgraph_batch(changed, batch, (0, 4, 8), 4)

    first, second, third = (slice(0, 4), slice(4, 8), slice(8, 12))
    assert torch.allclose(before[:, first], after[:, first])
    assert torch.allclose(before[:, third], after[:, third])
    assert not torch.allclose(before[:, second], after[:, second])


def test_gps_delta_reaches_every_parameter():
    torch.manual_seed(2)
    scorer = _scorer(layers=2, heads=2)
    hidden = [torch.randn(8, 4, dtype=torch.float64) for _ in range(2)]
    batch = next(scorer.graph_batches())
    scorer.score_subgraph_batch(hidden, batch, (0, 4), 4).sum().backward()
    # A branch multiplied by zero still produces a gradient tensor, so require
    # that something actually flowed.
    dead = [
        name
        for name, value in scorer.mixer.named_parameters()
        if value.grad is None or not value.grad.any()
    ]
    assert dead == []


def test_gps_refuses_whole_context_scoring():
    scorer = _scorer()
    with pytest.raises(ValueError, match="fixed-size subgraphs"):
        scorer(torch.randn(1, 8, 4, dtype=torch.float64), token_microbatch_size=8)


def test_gps_trainer_requires_a_subgraph_size():
    scorer = _scorer()
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer, gate_lr=1e-3, mixer_lr=1e-3
    )
    with pytest.raises(ValueError, match="fixed-size subgraphs"):
        GraphTrainer(
            scorer,
            gate_optimizer=gate_optimizer,
            mixer_optimizer=mixer_optimizer,
            token_microbatch_size=8,
        )


def test_gps_training_step_updates_every_mixer_parameter():
    """One joint subgraph update must move the whole GPS stack, not part of it."""

    torch.manual_seed(3)
    scorer = _scorer(layers=1, heads=1)
    # No weight decay: otherwise a decayed parameter moves even with a zero
    # gradient, and this test would pass on a branch that never learns.
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer, gate_lr=1e-2, mixer_lr=1e-2, weight_decay=0.0
    )
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        mixer_optimizer=mixer_optimizer,
        token_microbatch_size=8,
        subgraph_size=4,
    )
    before = {
        name: value.detach().clone()
        for name, value in scorer.mixer.named_parameters()
    }
    result = trainer.train_context(_example(tokens=8), mode="joint")

    assert result["mixer_steps"] > 0
    assert torch.isfinite(result["joint_loss"])
    unchanged = [
        name
        for name, value in scorer.mixer.named_parameters()
        if torch.equal(before[name], value.detach())
    ]
    assert unchanged == []


def test_gps_validation_scores_a_held_out_context():
    """Validation runs every epoch, so it must work for GPS, not only training."""

    torch.manual_seed(14)
    scorer = _scorer(layers=1, heads=1)
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer, gate_lr=1e-3, mixer_lr=1e-3
    )
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        mixer_optimizer=mixer_optimizer,
        token_microbatch_size=8,
        subgraph_size=4,
    )
    example = _example(tokens=8)
    result = trainer.evaluate_context(example)
    assert result.optimizer_steps == 0

    # Compare against the plain mean, so a misaligned slice cannot pass.
    batch = next(scorer.graph_batches())
    scores = torch.cat(
        [
            scorer.score_subgraph_batch(list(example.hidden_by_layer), batch, (start,), 4)
            for start in (0, 4)
        ],
        dim=1,
    )
    expected = torch.nn.functional.binary_cross_entropy(
        scores, example.teacher_scores.reshape(scorer.num_graphs, -1), reduction="mean"
    )
    assert torch.allclose(result.loss, expected)


def test_gps_rejoins_streamed_chunks_into_one_subgraph():
    """Chunked delivery must not change the result: the stack spans the span."""

    torch.manual_seed(15)
    mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8).double()
    hidden = torch.randn(1, 6, 4, dtype=torch.float64)
    whole = mixer.prepare(hidden, (0,)).delta
    chunked = mixer.prepare_from_chunks(
        iter([(0, hidden[:, :2]), (2, hidden[:, 2:])]),
        graph_ids=(0,),
        token_count=6,
        token_microbatch_size=2,
    ).delta
    assert torch.allclose(whole, chunked)

    with pytest.raises(ValueError, match="do not cover the complete span"):
        mixer.prepare_from_chunks(
            iter([(0, hidden[:, :2])]),
            graph_ids=(0,),
            token_count=6,
            token_microbatch_size=2,
        )


def test_gps_training_gradients_match_an_independent_autograd_reference():
    """The trainer's per-subgraph accumulation must equal one plain backward."""

    torch.manual_seed(4)
    scorer = _scorer(layers=1, heads=1)
    example = _example(tokens=8)

    # Reference first, so both passes differentiate the same parameter values:
    # the trainer steps its optimizer once its backward is done.
    batch = next(scorer.graph_batches())
    scores = scorer.score_subgraph_batch(
        list(example.hidden_by_layer), batch, (0, 4), 4
    )
    targets = example.teacher_scores.reshape(scorer.num_graphs, -1)
    loss = torch.nn.functional.binary_cross_entropy(
        scores, targets, reduction="sum"
    ) / (scorer.num_graphs * 2 * 4)
    loss.backward()
    reference = {
        name: value.grad.detach().clone()
        for name, value in scorer.mixer.named_parameters()
    }

    for parameter in scorer.parameters():
        parameter.grad = None
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer, gate_lr=1e-3, mixer_lr=1e-3
    )
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        mixer_optimizer=mixer_optimizer,
        token_microbatch_size=8,
        subgraph_size=4,
    )
    trainer.train_context(example, mode="joint")

    for name, value in scorer.mixer.named_parameters():
        assert torch.allclose(reference[name], value.grad, atol=1e-10), name


def test_gps_works_through_the_answer_training_scoring_and_replay_path():
    """The answer objective replays scores separately; GPS must survive that too."""

    from graph import replay_score_gradients, score_context_subgraphs

    torch.manual_seed(20)
    scorer = _scorer(layers=2, heads=2)
    hidden = [torch.randn(8, 4, dtype=torch.float64) for _ in range(2)]
    scores = score_context_subgraphs(
        scorer, hidden, subgraph_size=4, token_microbatch_size=8
    )
    assert tuple(scores.shape) == (2, 1, 2, 8)

    replay_score_gradients(
        scorer,
        hidden,
        torch.randn_like(scores),
        torch.zeros(2, 1, 2, 2, dtype=torch.long),
        subgraph_size=4,
        token_microbatch_size=8,
    )
    missing = [
        name for name, value in scorer.mixer.named_parameters() if value.grad is None
    ]
    assert missing == []

    with pytest.raises(ValueError, match="fixed-size subgraphs"):
        score_context_subgraphs(scorer, hidden, token_microbatch_size=8)


def test_each_graph_uses_its_own_weights():
    """Every per-graph weight must be selected by graph, not shared from one row.

    Sharing any of them still produces plausible scores, so only perturbing one
    graph's weights and watching the others stay put catches it.
    """

    torch.manual_seed(21)
    scorer = _scorer(layers=2, heads=2)
    hidden = [torch.randn(6, 4, dtype=torch.float64) for _ in range(2)]
    batch = next(scorer.graph_batches())
    assert len(batch.graph_ids) > 1
    before = scorer.score_subgraph_batch(hidden, batch, (0,), 6)

    # The perturbation has to be non-uniform: the block's normalizations are
    # invariant to a constant shift across a token's features, so adding one
    # would cancel and look like an unreachable weight.
    def perturbed(values):
        with torch.no_grad():
            saved = values.detach().clone()
            values[0] += torch.randn_like(values[0])
        try:
            return scorer.score_subgraph_batch(hidden, batch, (0,), 6)
        finally:
            with torch.no_grad():
                values.copy_(saved)

    for name, parameter in scorer.mixer.named_parameters():
        after = perturbed(parameter)
        assert not torch.allclose(before[0], after[0]), f"{name} never reached graph 0"
        assert torch.allclose(before[1:], after[1:]), f"{name} leaked out of graph 0"

    # Buffers are per-graph too: a shared random-feature draw is still plausible.
    after = perturbed(scorer.mixer.blocks[0].attention.projection)
    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1:], after[1:])


def test_each_attention_head_uses_its_own_random_features():
    torch.manual_seed(22)
    attention = _PerGraphPerformerAttention(1, 8, 2, 8).double()
    hidden = torch.randn(1, 6, 8, dtype=torch.float64)
    before = attention(hidden, (0,))
    with torch.no_grad():
        attention.projection[:, 1] = attention.projection[:, 0]
    assert not torch.allclose(before, attention(hidden, (0,)))


def test_a_real_gps_run_writes_a_checkpoint_the_evaluator_accepts(tmp_path):
    """Round-trip the config a training run actually builds, not a hand-made one."""

    import train_graph
    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(31)
    options = train_graph.resolve_options(
        _train_args(
            "--mixer-architecture", "gps",
            "--graph-dim", "4",
            "--gps-depth", "2",
            "--gps-attention-heads", "2",
            "--gps-random-features", "8",
            "--subgraph-size", "4",
            "--token-microbatch-size", "4",
            "--graph-microbatch-size", "1",
        )
    )
    scorer = _scorer(layers=1, heads=1)
    config = train_graph.normalized_checkpoint_config(
        model_id="unit", scorer=scorer, options=options, query_groups=1
    )

    assert config["mixer_architecture"] == "gps"
    assert config["activation_order"] == GPS_ACTIVATION_ORDER
    assert (config["gps_depth"], config["gps_attention_heads"]) == (2, 2)
    assert config["gps_random_features"] == 8

    # The evaluator must accept it as written, with no hand-editing.
    checkpoint = load_evaluation_checkpoint(_save(tmp_path, scorer, config))
    assert checkpoint.mixer_architecture == "gps"
    assert checkpoint.subgraph_size == 4


def test_a_real_implicit_run_still_writes_the_implicit_activation_order():
    import train_graph

    options = train_graph.resolve_options(_train_args("--graph-dim", "4"))
    config = train_graph.normalized_checkpoint_config(
        model_id="unit",
        scorer=_scorer(layers=1, heads=1, architecture="implicit"),
        options=options,
        query_groups=1,
    )
    assert config["mixer_architecture"] == "implicit"
    assert config["activation_order"] == mixer_activation_order("implicit")


def test_gps_checkpoint_rebuilds_into_a_scorer_that_reproduces_its_scores(tmp_path):
    """The evaluation path must rebuild GPS, not quietly rebuild an implicit mixer."""

    from graph.evaluation import load_evaluation_checkpoint, reconstruct_graph_scorer

    torch.manual_seed(23)
    scorer = _scorer(layers=1, heads=1)
    hidden = torch.randn(1, 4, 4, dtype=torch.float64)
    expected = scorer.mixer.delta(hidden, (0,))

    path = _save(tmp_path, scorer, _gps_checkpoint_config())
    checkpoint = load_evaluation_checkpoint(path)
    model = SimpleNamespace(
        config=_config(1, 1), device="cpu", gates=None, model=SimpleNamespace()
    )
    rebuilt = reconstruct_graph_scorer(checkpoint, model)

    assert rebuilt.mixer_architecture == "gps"
    assert rebuilt.mixer.depth == 2
    assert rebuilt.mixer.attention_heads == 2
    assert rebuilt.mixer.random_features == 8
    # The rebuilt gates are the real runtime gates, not this file's stand-ins,
    # so compare the mixer's own correction rather than the gate's scores.
    assert torch.allclose(expected, rebuilt.mixer.delta(hidden, (0,)))


def test_gps_scores_a_hidden_cache_the_way_the_evaluator_does(tmp_path):
    """This is the path eval_graph runs, and no other test reaches it.

    It differs from the training path: it runs under inference mode, and it
    offsets subgraph starts past a cached prefix.
    """

    from graph.evaluation import (
        load_evaluation_checkpoint,
        reconstruct_graph_scorer,
        score_hidden_cache,
    )

    torch.manual_seed(45)
    saved = _scorer(layers=1, heads=1)
    checkpoint = load_evaluation_checkpoint(
        _save(tmp_path, saved, _gps_checkpoint_config())
    )
    model = SimpleNamespace(
        config=_config(1, 1), device="cpu", gates=None, model=SimpleNamespace()
    )
    scorer = reconstruct_graph_scorer(checkpoint, model)

    prefix, tokens = 2, 8
    context = torch.randn(tokens, 4, dtype=torch.float64)
    cache = [
        torch.cat(
            (torch.zeros(1, prefix, 4, dtype=torch.float64), context.unsqueeze(0)), dim=1
        )
    ]
    with torch.inference_mode():
        scores = score_hidden_cache(
            scorer,
            cache,
            start_idx=prefix,
            end_idx=prefix + tokens,
            token_microbatch_size=4,
            subgraph_size=4,
        )
    assert tuple(scores.shape) == (1, 1, 1, tokens)
    assert torch.isfinite(scores).all()

    # Each subgraph must be scored on its own tokens, offset past the prefix.
    batch = next(scorer.graph_batches())
    expected = torch.cat(
        [scorer.score_subgraph_batch([context], batch, (start,), 4) for start in (0, 4)],
        dim=1,
    )
    assert torch.allclose(scores.reshape(1, -1), expected)

    # And the whole-context branch of the same entry point must refuse.
    with pytest.raises(ValueError, match="fixed-size subgraphs"):
        score_hidden_cache(
            scorer,
            cache,
            start_idx=prefix,
            end_idx=prefix + tokens,
            token_microbatch_size=4,
        )


def test_the_overflow_stabilizer_is_load_bearing():
    """Without it the exponential overflows on inputs the block will really see."""

    torch.manual_seed(24)
    attention = _PerGraphPerformerAttention(1, 16, 2, 16)
    hidden = torch.randn(1, 32, 16) * 30.0
    packed = attention.qkv_proj(hidden, (0,))
    query = packed.split(16, dim=-1)[0].view(1, 32, 2, 8) * attention.head_dim**-0.25

    stabilized = attention._features(query, attention.projection, per_token=True)
    scores = torch.einsum("gthd,ghmd->gthm", query, attention.projection)
    raw = torch.exp(scores - query.square().sum(dim=-1, keepdim=True) / 2)

    # Every feature of a token underflowing to zero makes that token's attention
    # denominator zero, so its output is decided by the division guard alone.
    assert int((raw.sum(-1) == 0).sum()) > 20
    assert int((stabilized.sum(-1) == 0).sum()) == 0
    assert torch.isfinite(attention(hidden, (0,))).all()


def test_prepared_tokens_keep_their_order_when_sliced():
    """Validation slices a prepared span; a reordered slice misaligns scores."""

    torch.manual_seed(25)
    mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8).double()
    prepared = mixer.prepare(torch.randn(1, 6, 4, dtype=torch.float64), (0,))
    index = torch.tensor([4, 1, 3])
    selected = prepared.select_tokens(index)
    for position, token in enumerate(index.tolist()):
        assert torch.equal(selected.delta[0, position], prepared.delta[0, token])


def test_layer_norm_actually_normalizes():
    torch.manual_seed(26)
    norm = PerGraphLayerNorm(1, 8).double()
    values = norm(torch.randn(1, 5, 8, dtype=torch.float64) * 7 + 3, (0,))
    assert torch.allclose(values.mean(-1), torch.zeros(1, 5, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(values.std(-1, unbiased=False), torch.ones(1, 5, dtype=torch.float64), atol=1e-4)


def test_the_block_is_wired_the_way_the_recipe_says():
    """Both branches, each normalized, then a feedforward with its activation.

    Compared against the whole computation written out, so dropping the
    activation, reordering the normalizations, or losing a branch all fail.
    """

    torch.manual_seed(27)
    block = _GPSBlock(
        1, 4, attention_heads=2, random_features=8, gram_normalization="token-count"
    ).double()
    hidden = torch.randn(1, 5, 4, dtype=torch.float64)

    local = block.local_norm(hidden + block.local(hidden, (0,)), (0,))
    attended = block.attention_norm(hidden + block.attention(hidden, (0,)), (0,))
    merged = local + attended
    expected = block.ffn_norm(
        merged + block.ffn_out(F.gelu(block.ffn_in(merged, (0,))), (0,)), (0,)
    )

    assert torch.allclose(block(hidden, (0,)), expected)
    # And the activation is not the identity in disguise.
    assert not torch.allclose(
        block.ffn_out(F.gelu(block.ffn_in(merged, (0,))), (0,)),
        block.ffn_out(block.ffn_in(merged, (0,)), (0,)),
    )


def test_the_final_activation_and_its_slope_are_applied():
    """The plan promises the same activation as the implicit mixer."""

    torch.manual_seed(28)
    steep = GPSGraphMixer(
        1, 4, 4, attention_heads=2, random_features=8, leaky_relu_slope=0.5
    ).double()
    shallow = GPSGraphMixer(
        1, 4, 4, attention_heads=2, random_features=8, leaky_relu_slope=0.01
    ).double()
    shallow.load_state_dict(steep.state_dict())

    hidden = torch.randn(1, 6, 4, dtype=torch.float64)
    steep_delta, shallow_delta = steep.delta(hidden, (0,)), shallow.delta(hidden, (0,))
    negative = shallow_delta < 0
    assert negative.any(), "no negative outputs, so the slope cannot be observed"
    assert not torch.allclose(steep_delta, shallow_delta)
    # Positive outputs pass through untouched; only the negative side scales.
    assert torch.allclose(steep_delta[~negative], shallow_delta[~negative])


def test_the_local_branch_is_the_implicit_aggregation():
    """The whole architecture comparison rests on this branch being the same.

    Give the implicit mixer an input already at graph width and an identity
    output projection. Its aggregation then reduces to exactly what the GPS
    local branch computes, so the two must agree on shared weights.
    """

    torch.manual_seed(40)
    width, tokens = 6, 9
    implicit = ImplicitGraphMixer(1, width, width, gram_normalization="token-count").double()
    branch = _PerGraphImplicitBranch(1, width, gram_normalization="token-count").double()
    with torch.no_grad():
        implicit.out_proj.weight.copy_(torch.eye(width, dtype=torch.float64).unsqueeze(0))
        branch.proj.weight.copy_(implicit.in_proj.weight)

    values = torch.randn(1, tokens, width, dtype=torch.float64)
    prepared = implicit.prepare(values, (0,), token_microbatch_size=tokens)
    expected = implicit._raw(prepared.y1, prepared.kernel)

    assert torch.allclose(expected, branch(values, (0,)))


def test_the_local_branch_normalizes_its_gram_by_the_token_count():
    torch.manual_seed(29)
    counted = _PerGraphImplicitBranch(1, 4, gram_normalization="token-count").double()
    raw = _PerGraphImplicitBranch(1, 4, gram_normalization="none").double()
    raw.load_state_dict(counted.state_dict())
    hidden = torch.randn(1, 5, 4, dtype=torch.float64)
    assert torch.allclose(counted(hidden, (0,)) * 5, raw(hidden, (0,)))


def test_weight_decay_skips_the_gps_normalizations_and_residual_weight():
    mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8)
    decay, no_decay = mixer.parameter_groups()
    decayed = {id(parameter) for parameter in decay}
    undecayed = {id(parameter) for parameter in no_decay}
    assert id(mixer.alpha) in undecayed
    for block in mixer.blocks:
        for norm in (block.local_norm, block.attention_norm, block.ffn_norm):
            assert id(norm.weight) in undecayed and id(norm.bias) in undecayed
        assert id(block.ffn_in.weight) in decayed
    assert decayed.isdisjoint(undecayed)
    assert len(decayed | undecayed) == len(list(mixer.parameters()))


def test_gps_delta_is_promoted_like_the_implicit_delta():
    """The gate reads hidden + delta; a bfloat16 delta breaks its normalization."""

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8).to(dtype)
        delta = mixer.delta(torch.randn(1, 5, 4, dtype=dtype), (0,))
        assert delta.dtype == torch.float32, dtype


def test_random_features_are_redrawn_on_the_chosen_schedule():
    """FAVOR+ resamples; one frozen draw is a distortion the model can fit."""

    torch.manual_seed(41)
    mixer = GPSGraphMixer(
        1, 4, 4, depth=2, attention_heads=2, random_features=8, redraw_interval=3
    ).double()
    mixer.train()
    first = [block.attention.projection.clone() for block in mixer.blocks]

    for step in (1, 2):
        mixer.on_optimizer_step()
        assert all(
            torch.equal(before, block.attention.projection)
            for before, block in zip(first, mixer.blocks)
        ), f"redrew early, at step {step}"

    mixer.on_optimizer_step()
    assert all(
        not torch.equal(before, block.attention.projection)
        for before, block in zip(first, mixer.blocks)
    ), "every block must get a fresh draw"


def test_random_features_are_never_redrawn_outside_training():
    torch.manual_seed(42)
    mixer = GPSGraphMixer(
        1, 4, 4, attention_heads=2, random_features=8, redraw_interval=1
    ).double()
    mixer.eval()
    before = mixer.blocks[0].attention.projection.clone()
    mixer.on_optimizer_step()
    assert torch.equal(before, mixer.blocks[0].attention.projection)
    assert int(mixer.redraw_step) == 0


def test_a_zero_interval_never_redraws():
    torch.manual_seed(43)
    mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8).double()
    mixer.train()
    before = mixer.blocks[0].attention.projection.clone()
    for _ in range(5):
        mixer.on_optimizer_step()
    assert torch.equal(before, mixer.blocks[0].attention.projection)


def test_the_redraw_schedule_survives_a_resume():
    """A resumed run must not restart its schedule from zero."""

    torch.manual_seed(44)
    mixer = GPSGraphMixer(
        1, 4, 4, attention_heads=2, random_features=8, redraw_interval=4
    ).double()
    mixer.train()
    mixer.on_optimizer_step()
    mixer.on_optimizer_step()
    assert int(mixer.redraw_step) == 2

    restored = GPSGraphMixer(
        1, 4, 4, attention_heads=2, random_features=8, redraw_interval=4
    ).double()
    restored.load_state_dict(mixer.state_dict())
    restored.train()
    assert int(restored.redraw_step) == 2
    before = restored.blocks[0].attention.projection.clone()
    restored.on_optimizer_step()
    restored.on_optimizer_step()
    assert not torch.equal(before, restored.blocks[0].attention.projection)


def test_the_default_interval_gives_about_thirty_redraws():
    """Copying the reference interval of 1000 would redraw nothing here."""

    import train_graph

    options = train_graph.resolve_options(
        _train_args(
            "--mixer-architecture", "gps",
            "--graph-dim", "4",
            "--gps-attention-heads", "2",
            "--subgraph-size", "4",
            "--token-microbatch-size", "4",
        )
    )
    for horizon in (930, 116, 96):
        interval = train_graph.resolve_redraw_interval(options, total_steps=horizon)
        assert interval >= 1
        assert 20 <= horizon // interval <= 40, (horizon, interval)

    # An explicit choice wins, and the implicit architecture never redraws.
    explicit = replace(options, gps_redraw_interval=7)
    assert train_graph.resolve_redraw_interval(explicit, total_steps=930) == 7
    implicit = replace(options, mixer_architecture="implicit")
    assert train_graph.resolve_redraw_interval(implicit, total_steps=930) == 0


def test_redraw_interval_is_refused_under_the_implicit_architecture():
    import train_graph

    with pytest.raises(ValueError, match="requires --mixer-architecture gps"):
        train_graph.resolve_options(_train_args("--gps-redraw-interval", "5"))


def test_gps_state_round_trips_including_its_random_features():
    """A reloaded checkpoint must reproduce the scores it was trained to give."""

    torch.manual_seed(5)
    scorer = _scorer(layers=1, heads=1)
    hidden = [torch.randn(8, 4, dtype=torch.float64)]
    batch = next(scorer.graph_batches())
    expected = scorer.score_subgraph_batch(hidden, batch, (0, 4), 4)

    torch.manual_seed(99)
    restored = _scorer(layers=1, heads=1)
    projection = "mixer.blocks.0.attention.projection"
    assert not torch.equal(
        scorer.state_dict()[projection], restored.state_dict()[projection]
    )
    restored.load_state_dict(scorer.state_dict(), strict=True)
    assert torch.equal(
        scorer.state_dict()[projection], restored.state_dict()[projection]
    )
    assert torch.allclose(
        expected, restored.score_subgraph_batch(hidden, batch, (0, 4), 4)
    )


def test_graph_dim_must_divide_across_the_attention_heads():
    with pytest.raises(ValueError, match="multiple of the head count"):
        GPSGraphMixer(1, 4, 6, attention_heads=4, random_features=8)


def test_position_encoding_distinguishes_repeated_tokens():
    """Identical tokens must still be told apart by their place in the subgraph."""

    torch.manual_seed(6)
    mixer = GPSGraphMixer(1, 4, 4, attention_heads=2, random_features=8).double()
    repeated = torch.ones(1, 6, 4, dtype=torch.float64)
    delta = mixer.delta(repeated, (0,))
    # Every position must differ from every other, not merely alternate: an
    # encoding that repeated with a short period would still pass a single pair.
    for first in range(6):
        for second in range(first + 1, 6):
            assert not torch.allclose(
                delta[0, first], delta[0, second]
            ), f"positions {first} and {second} are indistinguishable"


def test_the_sequence_position_encoding_is_well_formed():
    """Shape and distinctness only. That it reaches the stack is pinned by
    `test_position_encoding_distinguishes_repeated_tokens` above, which fails
    when the encoding stops being added."""

    torch.manual_seed(30)
    encoding = sinusoidal_positions(8, 6, device=None, dtype=torch.float64)
    assert encoding.shape == (8, 6)
    assert not torch.allclose(encoding[0], encoding[1])
    # An odd width must still be valid, since graph width is a free setting.
    assert sinusoidal_positions(4, 5, device=None, dtype=torch.float64).shape == (4, 5)


def _gps_checkpoint_config():
    return {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": 1,
        "gate_sink": 1,
        "hidden_dim": 4,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 4,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 4,
        "subgraph_size": 4,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": GPS_ACTIVATION_ORDER,
        "alpha_init": 0.1,
        "mixer_architecture": "gps",
        "gps_depth": 2,
        "gps_attention_heads": 2,
        "gps_random_features": 8,
        "gps_redraw_interval": 0,
    }


def _save(tmp_path, scorer, config):
    from graph import save_checkpoint

    return save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config=config,
        model_id="unit",
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor={"epoch": 0},
        wandb_run_id=None,
    )


def test_gps_checkpoint_validates_its_whole_stack(tmp_path):
    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(10)
    path = _save(tmp_path, _scorer(), _gps_checkpoint_config())
    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.mixer_architecture == "gps"
    assert checkpoint.subgraph_size == 4

    payload = torch.load(path, weights_only=False)
    assert "mixer.blocks.1.attention.projection" in payload["mixer"]
    payload["mixer"].pop("mixer.blocks.1.ffn_out.weight")
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="missing tensor"):
        load_evaluation_checkpoint(broken)


def test_gps_checkpoint_must_record_the_subgraph_size_it_trained_at(tmp_path):
    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(11)
    config = _gps_checkpoint_config()
    config.pop("subgraph_size")
    path = _save(tmp_path, _scorer(), config)
    with pytest.raises(ValueError, match="does not score whole contexts"):
        load_evaluation_checkpoint(path)


def test_gps_checkpoint_rejects_the_implicit_activation_order(tmp_path):
    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(12)
    config = _gps_checkpoint_config()
    config["activation_order"] = "batchnorm-leaky-relu"
    path = _save(tmp_path, _scorer(), config)
    with pytest.raises(ValueError, match="activation order"):
        load_evaluation_checkpoint(path)


def test_a_checkpoint_without_an_architecture_still_loads_as_implicit(tmp_path):
    """Checkpoints predating the choice carry no such key and are all implicit."""

    from graph.evaluation import load_evaluation_checkpoint

    torch.manual_seed(13)
    config = {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": 1,
        "gate_sink": 1,
        "hidden_dim": 4,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 4,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 4,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
    }
    path = _save(tmp_path, _scorer(architecture="implicit"), config)
    checkpoint = load_evaluation_checkpoint(path)
    assert "mixer_architecture" not in checkpoint.config
    assert checkpoint.mixer_architecture == "implicit"


def _train_args(*extra):
    import train_graph

    return train_graph.build_parser().parse_args(["--model", "unit", *extra])


def test_teacher_cli_refuses_gps_without_a_subgraph_size():
    import train_graph

    with pytest.raises(ValueError, match="requires --subgraph-size"):
        train_graph.resolve_options(_train_args("--mixer-architecture", "gps"))
    options = train_graph.resolve_options(
        _train_args(
            "--mixer-architecture",
            "gps",
            "--subgraph-size",
            "500",
            "--token-microbatch-size",
            "1000",
        )
    )
    assert options.mixer_architecture == "gps"
    assert options.subgraph_size == 500


def test_answer_cli_refuses_gps_without_a_subgraph_size():
    import train_graph_answer

    def resolve(*extra):
        return train_graph_answer.resolve_options(
            train_graph_answer.build_parser().parse_args(
                ["--model", "Qwen/unit", "--validation-retention-ratio", "0.2", *extra]
            )
        )

    with pytest.raises(ValueError, match="requires --subgraph-size"):
        resolve("--mixer-architecture", "gps")
    options = resolve(
        "--mixer-architecture",
        "gps",
        "--subgraph-size",
        "500",
        "--token-microbatch-size",
        "1000",
    )
    assert options.mixer_architecture == "gps"


def test_teacher_cli_rejects_a_graph_dim_the_heads_do_not_divide():
    import train_graph

    with pytest.raises(ValueError, match="multiple of --gps-attention-heads"):
        train_graph.resolve_options(
            _train_args(
                "--mixer-architecture",
                "gps",
                "--subgraph-size",
                "500",
                "--token-microbatch-size",
                "1000",
                "--graph-dim",
                "30",
                "--gps-attention-heads",
                "4",
            )
        )


def test_teacher_cli_defaults_to_the_implicit_architecture():
    import train_graph

    assert train_graph.resolve_options(_train_args()).mixer_architecture == "implicit"


def _answer_args(*extra):
    import train_graph_answer

    return train_graph_answer.build_parser().parse_args(
        ["--model", "Qwen/unit", "--validation-retention-ratio", "0.2", *extra]
    )


def _legacy_teacher_config():
    """A checkpoint config as written before the architecture became a choice."""

    return {
        "model_id": "unit",
        "compute_dtype": "bfloat16",
        "gate_dim": 16,
        "gate_sink": 16,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_kv_heads": 2,
        "query_groups": 1,
        "graph_dim": 32,
        "graph_microbatch_size": "auto",
        "token_microbatch_size": 1000,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
        "training_mode": "joint",
        "gate_lr": 1e-4,
        "mixer_lr": 1e-3,
        "adamw_eps": 1e-8,
        "amsgrad": False,
        "gate_lr_scheduler": None,
        "mixer_lr_scheduler": None,
        "freeze_gate": False,
        "epochs": 1,
        "seed": 0,
        "train_context_start": 0,
        "train_context_count": 29,
        "weight_decay": 0.01,
    }


def test_a_pre_change_checkpoint_still_resumes():
    """The new keys must not make every existing run's resume conflict."""

    import train_graph

    saved = _legacy_teacher_config()
    options = train_graph.resolve_options(
        _train_args(), {"config": saved, "model_id": "unit", "prefill_chunk": 4}
    )
    assert options.mixer_architecture == "implicit"

    # Resume compares the saved config against the one this run would write, so
    # the saved side must gain exactly the keys the new config emits. An
    # implicit run names its architecture and records no GPS settings.
    assert saved["mixer_architecture"] == "implicit"
    assert not [key for key in saved if key.startswith("gps_")]


def test_a_pre_change_answer_checkpoint_still_resumes():
    import train_graph_answer

    saved = train_graph_answer.normalized_answer_resume_config(
        {**_legacy_teacher_config(), "data": "agentic"}
    )
    assert saved["mixer_architecture"] == "implicit"
    assert not [key for key in saved if key.startswith("gps_")]


def _recorded_keys(graph_dim, architecture, **extra):
    """The architecture keys a run with these settings would write."""

    import train_graph

    flags = ["--graph-dim", str(graph_dim)] if graph_dim is not None else []
    if architecture == "gps":
        flags += [
            "--mixer-architecture", "gps",
            "--subgraph-size", "4",
            "--token-microbatch-size", "4",
            "--gps-attention-heads", "2",
        ]
    options = train_graph.resolve_options(_train_args(*flags))
    if graph_dim is None:
        options = replace(options, graph_dim=None)
    config = train_graph.normalized_checkpoint_config(
        model_id="unit",
        scorer=_scorer(layers=1, heads=1, architecture="implicit"),
        options=options,
        query_groups=1,
        **extra,
    )
    return {
        key
        for key in config
        if key == "mixer_architecture" or key.startswith("gps_")
    }


def test_a_gate_only_run_records_no_architecture_at_all():
    """No mixer means no architecture to name, and nothing to conflict on."""

    assert _recorded_keys(None, "implicit") == set()


def test_an_implicit_run_records_the_architecture_but_no_gps_settings():
    assert _recorded_keys(4, "implicit") == {"mixer_architecture"}


def test_a_gps_run_records_the_architecture_and_its_settings():
    assert _recorded_keys(4, "gps") == {
        "mixer_architecture",
        "gps_depth",
        "gps_attention_heads",
        "gps_random_features",
        "gps_redraw_interval",
    }


def _resume_agrees(saved_config, written_keys):
    """Whether resume would compare equal on the architecture keys."""

    import train_graph_answer

    back_filled = train_graph_answer.normalized_answer_resume_config(saved_config)
    present = {
        key
        for key in back_filled
        if key == "mixer_architecture" or key.startswith("gps_")
    }
    return present == written_keys


def test_an_old_gate_only_checkpoint_still_resumes():
    """The saved and written sides must agree, or every old run is stranded."""

    old = {**_legacy_teacher_config(), "graph_dim": None, "data": "agentic"}
    assert _resume_agrees(old, _recorded_keys(None, "implicit"))


def test_an_old_implicit_checkpoint_still_resumes():
    old = {**_legacy_teacher_config(), "data": "agentic"}
    assert _resume_agrees(old, _recorded_keys(4, "implicit"))


def test_resume_cannot_switch_the_architecture():
    import train_graph

    saved = {**_legacy_teacher_config(), "subgraph_size": 500}
    with pytest.raises(ValueError, match="mixer_architecture"):
        train_graph.resolve_options(
            _train_args("--mixer-architecture", "gps"),
            {"config": saved, "model_id": "unit", "prefill_chunk": 4},
        )


def test_answer_resume_cannot_switch_the_architecture():
    import train_graph_answer

    saved = {
        **_legacy_teacher_config(),
        "model_id": "Qwen/unit",
        "data": "agentic",
        "subgraph_size": 500,
        "validation_retention_ratio": 0.2,
    }
    # Warm-starting from existing graph weights locks the architecture, the same
    # way a full resume does.
    with pytest.raises(ValueError, match="mixer_architecture|conflict"):
        train_graph_answer.resolve_options(
            _answer_args(
                "--graph-checkpoint", "source.pt",
                "--mixer-architecture", "gps",
                "--subgraph-size", "500",
                "--token-microbatch-size", "1000",
            ),
            {"config": saved, "model_id": "Qwen/unit", "prefill_chunk": 4},
        )


def test_gps_settings_are_refused_under_the_implicit_architecture():
    """A setting nothing applies must never reach the checkpoint."""

    import train_graph

    for flag, value in (
        ("--gps-depth", "3"),
        ("--gps-attention-heads", "8"),
        ("--gps-random-features", "64"),
    ):
        with pytest.raises(ValueError, match="requires --mixer-architecture gps"):
            train_graph.resolve_options(_train_args(flag, value))


def test_answer_cli_refuses_gps_settings_under_the_implicit_architecture():
    import train_graph_answer

    for flag, value in (
        ("--gps-depth", "3"),
        ("--gps-attention-heads", "8"),
        ("--gps-random-features", "64"),
    ):
        with pytest.raises(ValueError, match="requires --mixer-architecture gps"):
            train_graph_answer.resolve_options(_answer_args(flag, value))


def test_answer_cli_rejects_a_graph_dim_the_heads_do_not_divide():
    import train_graph_answer

    with pytest.raises(ValueError, match="multiple of --gps-attention-heads"):
        train_graph_answer.resolve_options(
            _answer_args(
                "--mixer-architecture", "gps",
                "--subgraph-size", "500",
                "--token-microbatch-size", "1000",
                "--graph-dim", "30",
                "--gps-attention-heads", "4",
            )
        )


def test_a_gate_only_answer_run_rejects_the_gps_settings():
    import train_graph_answer

    for flag, value in (
        ("--mixer-architecture", "gps"),
        ("--gps-depth", "3"),
        ("--gps-attention-heads", "8"),
        ("--gps-random-features", "64"),
    ):
        with pytest.raises(ValueError, match="requires the graph mixer"):
            train_graph_answer.resolve_options(
                _answer_args("--no-graph-mixer", flag, value)
            )
