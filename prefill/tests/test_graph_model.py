import copy
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric import EdgeIndex

from attention.gate import Weight
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


def test_per_graph_linear_selects_reversed_noncontiguous_rows_and_gradients():
    linear = _model_symbol("PerGraphLinear")(
        num_graphs=4, in_features=1, out_features=1
    ).double()
    with torch.no_grad():
        linear.weight[:, 0, 0].copy_(
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        )
    x = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64, requires_grad=True)

    actual = linear(x, (3, 1))
    actual.sum().backward()

    torch.testing.assert_close(
        actual, torch.tensor([[[8.0]], [[6.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        linear.weight.grad[:, 0, 0],
        torch.tensor([0.0, 3.0, 0.0, 2.0], dtype=torch.float64),
    )


def _independent_gin(gin, z, edge_index, graph_ids, edge_weight):
    batch_size, token_count, dim = z.shape
    x = z.reshape(batch_size * token_count, dim)
    source, target = edge_index
    for depth, graph_mlps in enumerate(gin.mlps):
        messages = x[source] * edge_weight.unsqueeze(-1)
        aggregate = torch.zeros_like(x).index_add(0, target, messages)
        outputs = []
        for local_graph, graph_id in enumerate(graph_ids):
            start = local_graph * token_count
            stop = start + token_count
            combined = (
                (1 + gin.eps[depth, graph_id]) * x[start:stop]
                + aggregate[start:stop]
            )
            outputs.append(graph_mlps[graph_id](combined))
        x = torch.cat(outputs, dim=0)
    return x.view(batch_size, token_count, dim)


def test_depth_two_grouped_gin_matches_independent_outputs_and_all_gradients():
    torch.manual_seed(5)
    vectorized = _model_symbol("GroupedGIN")(num_graphs=4, dim=3, depth=2).double()
    independent = copy.deepcopy(vectorized)
    graph_ids = (3, 1)
    edge_index = EdgeIndex(
        torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]), sparse_size=(6, 6)
    )
    vectorized_z = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    independent_z = vectorized_z.detach().clone().requires_grad_(True)
    vectorized_weight = torch.randn(4, dtype=torch.float64, requires_grad=True)
    independent_weight = vectorized_weight.detach().clone().requires_grad_(True)
    output_weight = torch.randn(2, 3, 3, dtype=torch.float64)

    actual = vectorized(vectorized_z, edge_index, graph_ids, vectorized_weight)
    expected = _independent_gin(
        independent, independent_z, edge_index, graph_ids, independent_weight
    )
    (actual * output_weight).sum().backward()
    (expected * output_weight).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        vectorized_z.grad, independent_z.grad, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        vectorized_weight.grad, independent_weight.grad, rtol=1e-12, atol=1e-12
    )
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        vectorized.named_parameters(), independent.named_parameters()
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=1e-12,
            atol=1e-12,
        )


def test_grouped_gin_executes_two_batched_linears_per_depth(monkeypatch):
    gin = _model_symbol("GroupedGIN")(num_graphs=3, dim=2, depth=2).double()
    edge_index = EdgeIndex(
        torch.tensor([[0, 2, 4], [1, 3, 5]]), sparse_size=(6, 6)
    )
    calls = []
    original_bmm = torch.bmm

    def record_bmm(left, right):
        calls.append((left.shape, right.shape))
        return original_bmm(left, right)

    monkeypatch.setattr(torch, "bmm", record_bmm)

    gin(torch.randn(3, 2, 2, dtype=torch.float64), edge_index, (0, 1, 2))

    assert len(calls) == 4
    assert all(left_shape[0] == 3 for left_shape, _ in calls)


def test_grouped_gin_preserves_legacy_parameter_layout_and_strict_loading():
    source = _model_symbol("GroupedGIN")(num_graphs=2, dim=2, depth=2)
    expected_keys = {
        "eps",
        *{
            f"mlps.{depth}.{graph}.{linear}.{kind}"
            for depth in range(2)
            for graph in range(2)
            for linear in (0, 2)
            for kind in ("weight", "bias")
        },
    }
    target = _model_symbol("GroupedGIN")(num_graphs=2, dim=2, depth=2)

    assert set(source.state_dict()) == expected_keys
    target.load_state_dict(source.state_dict(), strict=True)


def test_differentiable_edge_weights_receive_gradients():
    class LearnableTestBuilder(GraphBuilder):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))

        def forward(self, z, k):
            assert k == 16
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
    topology = graph_builder(z, 16)

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
        [
            gate((hidden + delta[head]).unsqueeze(0))[0, head]
            for head in range(gate.nhead)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_headwise_gate_adapter_is_exact_for_real_bfloat16_gate_with_distinct_deltas():
    torch.manual_seed(41)
    gate = Weight(
        index=0,
        input_dim=4,
        output_dim=2,
        nhead=2,
        ngroup=3,
        dtype=torch.bfloat16,
        sink=2,
    )
    adapter = _model_symbol("_HeadwiseGateAdapter")()
    hidden = torch.randn(7, 4, dtype=torch.bfloat16)
    delta = torch.randn(gate.nhead, 7, 4, dtype=torch.bfloat16)

    actual = torch.stack(
        [adapter(gate, head, hidden, delta[head]) for head in range(gate.nhead)]
    )
    expected = torch.stack(
        [gate((hidden + delta[head]).unsqueeze(0))[0, head] for head in range(gate.nhead)]
    )

    assert torch.equal(actual, expected)


def test_low_precision_scorer_keeps_fp32_masters_and_zero_b_gate_bit_parity():
    torch.manual_seed(42)
    released_gate = Weight(
        index=0,
        input_dim=4,
        output_dim=2,
        nhead=2,
        ngroup=3,
        dtype=torch.bfloat16,
        sink=2,
    )
    baseline_gate = copy.deepcopy(released_gate)
    scorer = _model_symbol("GraphScorer")(
        [released_gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=2),
        graph_dim=2,
        graph_builder=FaissGraphBuilder(nlist=8),
        num_neighbors=1,
    )
    hidden = torch.randn(1, 7, 4, dtype=torch.bfloat16)

    actual = scorer(hidden)
    expected = baseline_gate(hidden)

    assert scorer.compute_dtype == torch.bfloat16
    assert torch.equal(actual, expected.unsqueeze(0))
    assert all(parameter.dtype == torch.float32 for parameter in scorer.parameters())
    assert scorer.gates[0].q_norm.weight.dtype == torch.float32
    assert scorer.gates[0].k_base.dtype == torch.float32


def test_explicit_compute_dtype_override_survives_fp32_reconstruction_shell():
    gate = Weight(0, 2, 1, 1, 1, torch.float32, sink=1)
    scorer = _model_symbol("GraphScorer")(
        [gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=1,
        compute_dtype=torch.bfloat16,
    )

    scores = scorer(torch.randn(1, 2, 2, dtype=torch.float32))

    assert scorer.compute_dtype == torch.bfloat16
    assert scorer.a_proj.weight.dtype == torch.float32
    assert scores.shape == (1, 1, 1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_precision_scorer_parameters_retain_compute_dtype(dtype):
    gate = Weight(0, 2, 1, 1, 1, dtype, sink=1).to(dtype=dtype)
    scorer = _model_symbol("GraphScorer")(
        [gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=1,
    )

    assert scorer.compute_dtype == dtype
    assert all(parameter.dtype == dtype for parameter in scorer.parameters())


def test_low_precision_graph_activations_stay_in_compute_dtype_and_builder_gets_k():
    class RecordingBuilder(GraphBuilder):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, z, k):
            self.calls.append((z.dtype, z.shape, k))
            token_count = z.size(1)
            edge_index = EdgeIndex(
                torch.empty((2, 0), dtype=torch.long, device=z.device),
                sparse_size=(z.size(0) * token_count, z.size(0) * token_count),
            )
            return GraphTopology(edge_index)

    builder = RecordingBuilder()
    gate = Weight(0, 4, 2, 1, 1, torch.bfloat16, sink=2)
    scorer = _model_symbol("GraphScorer")(
        [gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=3,
        graph_builder=builder,
        num_neighbors=7,
    )
    z = scorer.project_graph_nodes(
        torch.randn(1, 11, 4, dtype=torch.float32), (0,)
    )
    u = scorer.propagate_graph_nodes(z, (0,))

    assert z.shape == (1, 11, 3)
    assert u.shape == (1, 11, 3)
    assert z.dtype == u.dtype == torch.bfloat16
    assert builder.calls == [(torch.bfloat16, torch.Size([1, 11, 3]), 7)]


def test_low_precision_learnable_builder_is_fp32_master_and_receives_gradient():
    class WeightedBuilder(GraphBuilder):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.5]))

        def forward(self, z, k):
            assert k == 1
            edge_index = EdgeIndex(
                torch.tensor([[0, 1], [1, 0]], device=z.device),
                sparse_size=(2, 2),
            )
            return GraphTopology(edge_index, self.weight.expand(2))

    builder = WeightedBuilder()
    gate = Weight(0, 2, 1, 1, 1, torch.bfloat16, sink=1)
    scorer = _model_symbol("GraphScorer")(
        [gate],
        SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1),
        graph_dim=1,
        graph_builder=builder,
        num_neighbors=1,
    )
    with torch.no_grad():
        scorer.b_proj.weight.fill_(0.25)

    scorer(torch.randn(1, 2, 2, dtype=torch.bfloat16)).sum().backward()

    assert builder.weight.device == scorer.a_proj.weight.device
    assert builder.weight.dtype == torch.float32
    assert builder.weight.grad is not None
    assert builder.weight.grad.dtype == torch.float32
    assert torch.count_nonzero(builder.weight.grad) > 0


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
        graph_builder=FaissGraphBuilder(nlist=8),
        num_neighbors=1,
    ).double()
    return scorer, gates


def test_graph_batches_are_immutable_python_only_control_metadata():
    scorer, _ = _make_scorer()

    batches = list(scorer.graph_batches(microbatch_size=3))

    assert [batch.graph_ids for batch in batches] == [(0, 1, 2), (3,)]
    assert [batch.layer_ids for batch in batches] == [(0, 0, 1), (1,)]
    assert [batch.head_ids for batch in batches] == [(0, 1, 0), (1,)]
    assert all(
        isinstance(value, int)
        for batch in batches
        for values in (batch.graph_ids, batch.layer_ids, batch.head_ids)
        for value in values
    )
    with pytest.raises(FrozenInstanceError):
        batches[0].graph_ids = (0,)
    with pytest.raises(TypeError):
        list(scorer.graph_batches(device=torch.device("cpu")))

    graph_batch = _model_symbol("GraphBatch")
    with pytest.raises(TypeError, match="tuples of Python integers"):
        graph_batch([0], (0,), (0,))


def test_low_level_graph_identity_rejects_non_cpu_tensor_before_scalarization():
    linear = _model_symbol("PerGraphLinear")(2, 1, 1)
    non_cpu_ids = torch.empty(1, dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="CPU"):
        linear(torch.ones(1, 1, 1), non_cpu_ids)


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
