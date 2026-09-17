import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from graph import (
    GPS_ACTIVATION_ORDER,
    GPSGraphMixer,
    GraphTrainer,
    ImplicitGraphScorer,
    TeacherExample,
    build_adamw_optimizers,
    mixer_activation_order,
)
from graph.model import _PerGraphPerformerAttention, orthogonal_random_features


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
    missing = [
        name for name, value in scorer.mixer.named_parameters() if value.grad is None
    ]
    assert missing == []


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
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer, gate_lr=1e-2, mixer_lr=1e-2
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
    result = trainer.evaluate_context(_example(tokens=8))
    assert torch.isfinite(result.loss)
    assert result.optimizer_steps == 0


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


def test_gps_records_its_own_activation_order():
    assert mixer_activation_order("gps") == GPS_ACTIVATION_ORDER
    assert mixer_activation_order("implicit") != GPS_ACTIVATION_ORDER


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
    assert not torch.allclose(delta[0, 0], delta[0, 3])


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
