import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric import EdgeIndex

from graph.builder import FaissGraphBuilder, GraphBuilder, GraphTopology

try:
    import graph.model as model_module
except ImportError:
    model_module = None


class ReferenceGate(nn.Module):
    """Small runtime-gate equivalent with independently written score math."""

    def __init__(self, hidden_dim=3, heads=2, groups=2, gate_dim=2):
        super().__init__()
        self.nhead = heads
        self.ngroup = groups
        self.output_dim = gate_dim
        self.sink = 2
        self.d = math.sqrt(gate_dim)
        self.q_proj = nn.Linear(hidden_dim, heads * groups * gate_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, heads * gate_dim, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.k_base = nn.Parameter(torch.randn(heads, 1, self.sink, gate_dim))
        self.b = nn.Parameter(torch.randn(heads, 1, groups))

    def forward(self, hidden_states):
        x = hidden_states.squeeze(0)
        token_count = x.size(0)
        queries = self.q_proj(x).view(
            token_count, self.nhead, self.ngroup, self.output_dim
        )
        keys = self.k_proj(x).view(token_count, self.nhead, self.output_dim)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        logits = torch.einsum("thr,thgr->thg", keys, queries) / self.d
        logits = logits + self.b[:, 0].unsqueeze(0)
        base_logits = (
            torch.einsum("hsr,thgr->thsg", self.k_base[:, 0], queries) / self.d
        )
        scores = 1 / (
            1 + torch.exp(base_logits - logits.unsqueeze(2)).sum(dim=2)
        )
        return scores.mean(dim=-1).transpose(0, 1).unsqueeze(0)


def _model_symbol(name):
    assert model_module is not None, "graph.model is not implemented"
    symbol = getattr(model_module, name, None)
    assert symbol is not None, f"{name} is not implemented"
    return symbol


def _set_linear(linear, weight_scale):
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2, dtype=linear.weight.dtype) * weight_scale)
        linear.bias.zero_()


def test_grouped_gin_matches_two_independent_hand_computed_graphs():
    gin = _model_symbol("GroupedGIN")(num_graphs=2, dim=2, depth=1).double()
    _set_linear(gin.mlps[0][0][0], 1.0)
    _set_linear(gin.mlps[0][0][2], 1.0)
    _set_linear(gin.mlps[0][1][0], 2.0)
    _set_linear(gin.mlps[0][1][2], 1.0)
    with torch.no_grad():
        gin.eps[0].copy_(torch.tensor([0.0, 1.0], dtype=torch.float64))
    z = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float64,
    )
    edges = EdgeIndex(torch.tensor([[0, 2], [1, 3]]), sparse_size=(4, 4))

    actual = gin(z, edges, torch.tensor([0, 1]))

    expected = torch.tensor(
        [[[1.0, 2.0], [4.0, 6.0]], [[20.0, 24.0], [38.0, 44.0]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected)


def test_differentiable_edge_weights_receive_gradients():
    class LearnableTestBuilder(GraphBuilder):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))

        def forward(self, z):
            edge_index = EdgeIndex(torch.tensor([[0], [1]]), sparse_size=(2, 2))
            return GraphTopology(edge_index, self.weight)

    graph_builder = LearnableTestBuilder()
    assert list(graph_builder.parameters()) == [graph_builder.weight]
    gin = _model_symbol("GroupedGIN")(num_graphs=1, dim=1, depth=1).double()
    with torch.no_grad():
        gin.eps.zero_()
        for linear in (gin.mlps[0][0][0], gin.mlps[0][0][2]):
            linear.weight.fill_(1)
            linear.bias.zero_()
    z = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)
    topology = graph_builder(z)

    gin(z, topology.edge_index, torch.tensor([0]), topology.edge_weight).sum().backward()

    torch.testing.assert_close(
        graph_builder.weight.grad, torch.tensor([1.0], dtype=torch.float64)
    )


def test_headwise_gate_adapter_matches_full_gate_for_each_head_input():
    torch.manual_seed(3)
    gate = ReferenceGate().double()
    adapter = _model_symbol("_HeadwiseGateAdapter")()
    hidden = torch.randn(5, 3, dtype=torch.float64)
    delta = torch.randn(2, 5, 3, dtype=torch.float64)

    actual = torch.stack(
        [adapter(gate, head, hidden, delta[head]) for head in range(gate.nhead)]
    )
    expected = torch.stack(
        [gate((hidden + delta[head]).unsqueeze(0))[0, head] for head in range(gate.nhead)]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("value", "layers", "heads", "expected"),
    [("auto", 3, 4, 4), (1, 3, 4, 1), (12, 3, 4, 12)],
)
def test_graph_microbatch_resolution(value, layers, heads, expected):
    resolve = _model_symbol("resolve_graph_microbatch_size")
    assert resolve(value, layers, heads) == expected


@pytest.mark.parametrize("value", [0, 7, -1, 1.5, "bad", True])
def test_graph_microbatch_rejects_values_outside_flattened_graph_range(value):
    resolve = _model_symbol("resolve_graph_microbatch_size")
    with pytest.raises(ValueError):
        resolve(value, 2, 3)


def _make_scorer():
    torch.manual_seed(11)
    gates = [ReferenceGate().double(), ReferenceGate().double()]
    config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2)
    scorer = _model_symbol("GraphScorer")(
        gates,
        config,
        graph_dim=2,
        gin_depth=1,
        graph_builder=FaissGraphBuilder(k=1, nlist=8),
    ).double()
    return scorer, gates


def test_zero_b_returns_runtime_gate_baseline_and_expected_shape():
    scorer, gates = _make_scorer()
    hidden = torch.randn(2, 5, 3, dtype=torch.float64)

    scores = scorer(hidden, microbatch_size="auto")

    expected = torch.stack(
        [gates[layer](hidden[layer].unsqueeze(0)) for layer in range(2)]
    )
    assert scores.shape == (2, 1, 2, 5)
    torch.testing.assert_close(scores, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        scorer.last_delta_energy_share, torch.tensor(0.0, dtype=torch.float64)
    )
    assert scorer.a_proj.bias is None
    assert scorer.b_proj.bias is None
    assert scorer.num_layers == 2
    assert scorer.num_heads == 2
    assert scorer.hidden_dim == 3


def test_graph_microbatches_are_equivalent_and_hard_topology_keeps_live_z_gradient():
    scorer, _ = _make_scorer()
    with torch.no_grad():
        scorer.b_proj.weight.normal_(std=0.2)
    hidden = torch.randn(2, 6, 3, dtype=torch.float64)

    one_at_a_time = scorer(hidden, microbatch_size=1)
    energy_one = scorer.last_delta_energy_share.clone()
    all_at_once = scorer(hidden, microbatch_size=4)
    energy_all = scorer.last_delta_energy_share.clone()

    torch.testing.assert_close(one_at_a_time, all_at_once, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(energy_one, energy_all, rtol=1e-12, atol=1e-12)
    assert 0 <= energy_all.item() <= 1
    all_at_once.sum().backward()
    assert scorer.a_proj.weight.grad is not None
    assert torch.count_nonzero(scorer.a_proj.weight.grad) > 0
