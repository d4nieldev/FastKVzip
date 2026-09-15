"""Gate-only scoring must equal the plain FastKVzip gate it is fine-tuned from."""

from types import SimpleNamespace

import pytest
import torch

from attention.gate import Weight, load_fastkvzip
from graph.answer_training import score_context_subgraphs
from graph.model import ImplicitGraphMixer, ImplicitGraphScorer
from graph.training import save_checkpoint

LAYERS, HEADS, GROUPS, HIDDEN, GATE_DIM, SINK = 3, 2, 2, 8, 4, 3


def _config():
    return SimpleNamespace(
        num_hidden_layers=LAYERS,
        num_key_value_heads=HEADS,
        num_attention_heads=HEADS * GROUPS,
        hidden_size=HIDDEN,
    )


def _gates(dtype=torch.float32):
    torch.manual_seed(11)
    gates = []
    for layer in range(LAYERS):
        gate = Weight(layer, HIDDEN, GATE_DIM, HEADS, GROUPS, dtype, sink=SINK)
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.copy_(torch.randn_like(parameter, dtype=torch.float32))
        gates.append(gate)
    return gates


def _hidden(tokens=6):
    torch.manual_seed(5)
    return torch.randn(LAYERS, tokens, HIDDEN)


def _reference(gates, hidden):
    """Score every layer with the gate's own forward, as the evaluator does."""

    return torch.stack([gate(hidden[layer].unsqueeze(0))[0] for layer, gate in enumerate(gates)])


def test_gate_only_scorer_matches_the_plain_gate_forward():
    gates, hidden = _gates(), _hidden()
    scorer = ImplicitGraphScorer(gates, _config(), graph_dim=None)

    assert scorer.mixer is None
    assert scorer.graph_dim is None
    assert scorer.device == gates[0].q_proj.weight.device

    scores = scorer(hidden, token_microbatch_size=4)
    assert scores.shape == (LAYERS, 1, HEADS, hidden.size(1))
    torch.testing.assert_close(scores[:, 0], _reference(gates, hidden))


def test_gate_only_scorer_is_the_zero_alpha_limit_of_a_mixer_scorer():
    hidden = _hidden()
    bare = ImplicitGraphScorer(_gates(), _config(), graph_dim=None)
    mixed = ImplicitGraphScorer(_gates(), _config(), graph_dim=3, alpha_init=0.0)

    torch.testing.assert_close(
        bare(hidden, token_microbatch_size=4),
        mixed(hidden, token_microbatch_size=4),
    )


@pytest.mark.parametrize("subgraph_size", [None, 2, 3])
def test_subgraph_size_cannot_change_gate_only_scores(subgraph_size):
    gates, hidden = _gates(), _hidden()
    scorer = ImplicitGraphScorer(gates, _config(), graph_dim=None)

    scores = score_context_subgraphs(
        scorer, tuple(hidden), subgraph_size=subgraph_size, token_microbatch_size=6
    )
    torch.testing.assert_close(scores[:, 0], _reference(gates, hidden))


def test_gate_only_scorer_holds_the_mixer_arms_precision():
    """Dropping the mixer must not silently drop the gate to the compute dtype.

    With a mixer the delta is accumulated in the master dtype and promotes the
    gate input; a gate-only scorer has no delta, so it must materialize the
    hidden states in that same dtype or the ablation would also be a precision
    change.
    """

    gates = _gates(dtype=torch.bfloat16)
    scorer = ImplicitGraphScorer(gates, _config(), graph_dim=None)

    assert scorer.compute_dtype == torch.bfloat16
    assert scorer.hidden_dtype == torch.float32
    assert scorer(_hidden(), token_microbatch_size=4).dtype == torch.float32


def test_gate_only_scorer_keeps_no_mixer_state():
    scorer = ImplicitGraphScorer(_gates(), _config(), graph_dim=None)
    assert all(name.startswith("gates.") for name in scorer.state_dict())
    assert "mixer" not in dict(scorer.named_modules())


def test_gate_only_scoring_runs_none_of_the_mixer_math(monkeypatch):
    """Skipping the mixer, not zeroing it, is the point: no Gram, no kernel.

    The Gram matrices and the streamed context normalization are what dominate
    the step's memory, so a gate-only run must not compute them at all.
    """

    scorer = ImplicitGraphScorer(_gates(), _config(), graph_dim=None)
    prepared = []
    monkeypatch.setattr(
        ImplicitGraphMixer,
        "prepare_from_chunks",
        lambda *a, **k: prepared.append(1),
    )
    monkeypatch.setattr(ImplicitGraphMixer, "delta", lambda *a, **k: prepared.append(1))

    scorer(_hidden(), token_microbatch_size=4)
    assert prepared == []
    assert scorer.prepare(_hidden(), (0,), token_microbatch_size=4) is None


def test_gate_only_scorer_rejects_a_prepared_mixer_state():
    scorer = ImplicitGraphScorer(_gates(), _config(), graph_dim=None)
    with pytest.raises(ValueError, match="prepared mixer state"):
        scorer.score_prepared(
            torch.zeros(1, 2, HIDDEN), object(), layer_ids=(0,), head_ids=(0,)
        )


def test_a_trained_gate_reloads_through_the_evaluator_and_reproduces_its_scores(tmp_path):
    """The whole point of gate-only training: eval_chunk.py must see the same gate.

    Covers layer ordering, sink inference from the weights, and the precision
    the checkpoint was saved in.
    """

    gates, hidden = _gates(dtype=torch.bfloat16), _hidden()
    scorer = ImplicitGraphScorer(gates, _config(), graph_dim=None)
    expected = scorer(hidden, token_microbatch_size=4)

    path = save_checkpoint(
        tmp_path,
        "best",
        scorer=scorer,
        config={"compute_dtype": "bfloat16", "graph_dim": None},
        model_id="Qwen/unit",
        prefix_ids=torch.zeros(1, 1, dtype=torch.long),
        prefill_chunk=16,
        data_cursor={},
        wandb_run_id=None,
    )

    reloaded = load_fastkvzip("Qwen/unit", str(path), device="cpu")
    assert len(reloaded) == LAYERS
    assert all(gate.sink == SINK for gate in reloaded)
    assert all(gate.output_dim == GATE_DIM for gate in reloaded)
    # Training saved fp32 master weights, so the evaluator scores in fp32 too.
    assert reloaded[0].q_proj.weight.dtype == torch.float32

    torch.testing.assert_close(_reference(reloaded, hidden), expected[:, 0])


def test_layer_order_survives_a_checkpoint_with_ten_or_more_layers(tmp_path):
    """Sorting "0."…"11." as text would assign every gate to the wrong layer."""

    layers = 12
    config = SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=HEADS,
        num_attention_heads=HEADS * GROUPS,
        hidden_size=HIDDEN,
    )
    torch.manual_seed(3)
    gates = []
    for layer in range(layers):
        gate = Weight(layer, HIDDEN, GATE_DIM, HEADS, GROUPS, torch.float32, sink=SINK)
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.copy_(torch.randn_like(parameter))
        gates.append(gate)
    scorer = ImplicitGraphScorer(gates, config, graph_dim=None)

    path = save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config={"compute_dtype": "float32", "graph_dim": None},
        model_id="Qwen/unit",
        prefix_ids=torch.zeros(1, 1, dtype=torch.long),
        prefill_chunk=16,
        data_cursor={},
        wandb_run_id=None,
    )

    reloaded = load_fastkvzip("Qwen/unit", str(path), device="cpu")
    for original, loaded in zip(gates, reloaded):
        torch.testing.assert_close(original.q_proj.weight, loaded.q_proj.weight)
