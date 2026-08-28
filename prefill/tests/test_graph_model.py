import copy
import math
from types import SimpleNamespace

import torch
from torch import nn

from graph.model import (
    GraphBatch,
    ImplicitGraphMixer,
    ImplicitGraphScorer,
    PerGraphLinear,
    _HeadwiseGateAdapter,
    resolve_graph_microbatch_size,
)


class ReferenceGate(nn.Module):
    def __init__(self, hidden_dim=3, heads=2, groups=1, gate_dim=2):
        super().__init__()
        self.nhead = heads
        self.ngroup = groups
        self.output_dim = gate_dim
        self.sink = 1
        self.d = math.sqrt(gate_dim)
        self.q_proj = nn.Linear(hidden_dim, heads * groups * gate_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, heads * gate_dim, bias=False)
        self.q_norm = ReferenceRMSNorm(gate_dim)
        self.k_norm = ReferenceRMSNorm(gate_dim)
        self.k_base = nn.Parameter(torch.randn(heads, 1, self.sink, gate_dim))
        self.b = nn.Parameter(torch.randn(heads, 1, groups))

    def forward(self, hidden_states):
        x = hidden_states.squeeze(0)
        tokens = x.size(0)
        queries = self.q_proj(x).view(tokens, self.nhead, self.ngroup, self.output_dim)
        keys = self.k_proj(x).view(tokens, self.nhead, self.output_dim)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        logits = torch.einsum("thr,thgr->thg", keys, queries) / self.d
        logits = logits + self.b[:, 0].unsqueeze(0)
        base = torch.einsum("hsr,thgr->thsg", self.k_base[:, 0], queries) / self.d
        return (1 / (1 + torch.exp(base - logits.unsqueeze(2)).sum(2))).mean(-1).transpose(0, 1).unsqueeze(0)


class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        values = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * values.to(dtype)


def _config(layers=2, heads=2, hidden=3):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        num_attention_heads=heads,
        hidden_size=hidden,
    )


def test_implicit_multiplication_matches_explicit_dense_norm_then_activation_formula():
    mixer = ImplicitGraphMixer(1, 3, 2, gram_normalization="token-count").double()
    with torch.no_grad():
        mixer.in_proj.weight.copy_(
            torch.tensor(
                [[[1, 0, 0], [0, 1, 0], [1, -1, 0], [0, 0, 1]]],
                dtype=torch.float64,
            )
        )
        mixer.out_proj.weight.copy_(
            torch.tensor([[[1, 0], [0, 1], [1, -1]]], dtype=torch.float64)
        )
        mixer.gamma.fill_(1.5)
        mixer.beta.fill_(0.25)
        mixer.alpha.fill_(0.4)
    x = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [0.5, 1.5, -2.0]]],
        dtype=torch.float64,
    )
    actual = mixer(x, (0,))
    packed = torch.bmm(x, mixer.in_proj.weight.transpose(1, 2))
    y1, y2 = packed.split(2, dim=-1)
    adjacency = torch.bmm(y1, y1.transpose(1, 2))
    preactivation = (
        torch.bmm(torch.bmm(adjacency, y2), mixer.out_proj.weight.transpose(1, 2))
        / x.size(1)
    )
    mean = preactivation.mean(1, keepdim=True)
    variance = (preactivation - mean).square().mean(1, keepdim=True)
    normalized = (preactivation - mean) / torch.sqrt(variance + 1e-5)
    expected = mixer.alpha.view(1, 1, 1) * torch.nn.functional.leaky_relu(
        mixer.gamma.unsqueeze(1) * normalized + mixer.beta.unsqueeze(1),
        negative_slope=0.01,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_gram_normalization_modes_are_scaled_only_by_token_count():
    torch.manual_seed(1)
    x = torch.randn(1, 5, 3, dtype=torch.float64)
    scaled = ImplicitGraphMixer(1, 3, 2, gram_normalization="token-count").double()
    unscaled = ImplicitGraphMixer(1, 3, 2, gram_normalization="none").double()
    unscaled.load_state_dict(scaled.state_dict())
    a = scaled.prepare(x, (0,), token_microbatch_size=2)
    b = unscaled.prepare(x, (0,), token_microbatch_size=3)
    torch.testing.assert_close(b.gram, a.gram * x.size(1), rtol=1e-12, atol=1e-12)


def test_packed_w1_w2_slices_have_independent_gradients():
    linear = ImplicitGraphMixer(1, 2, 1).double()
    x = torch.tensor([[[2.0, -3.0]]], dtype=torch.float64)
    first, second = linear.project(x, (0,))
    first.sum().backward(retain_graph=True)
    assert linear.in_proj.weight.grad[:, 1:].abs().sum() == 0
    linear.in_proj.weight.grad.zero_()
    second.sum().backward()
    assert linear.in_proj.weight.grad[:, :1].abs().sum() == 0


def test_context_batchnorm_is_independent_per_graph_and_handles_degenerate_contexts():
    mixer = ImplicitGraphMixer(2, 2, 1).double()
    with torch.no_grad():
        mixer.in_proj.weight.fill_(1)
        mixer.out_proj.weight.fill_(1)
        mixer.beta.fill_(2)
        mixer.alpha.fill_(0.5)
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[11.0, 12.0], [13.0, 14.0]]],
        dtype=torch.float64,
    )
    prepared = mixer.prepare(x, (0, 1), token_microbatch_size=1)
    raw = mixer._raw(prepared.y1, prepared.kernel)
    normalized = mixer.normalized(raw, prepared)
    torch.testing.assert_close(normalized.mean(1), torch.zeros_like(normalized.mean(1)), atol=1e-12, rtol=0)
    constant = torch.ones(1, 1, 2, dtype=torch.float64)
    output = ImplicitGraphMixer(1, 2, 1).double()(constant, (0,))
    assert torch.isfinite(output).all()


def test_alpha_is_an_unconstrained_residual_scalar_with_gradient():
    mixer = ImplicitGraphMixer(1, 2, 1).double()
    with torch.no_grad():
        mixer.beta.fill_(1)
        mixer.alpha.fill_(-0.5)
    output = mixer(torch.randn(1, 3, 2, dtype=torch.float64), (0,))
    output.sum().backward()
    assert mixer.alpha.item() == -0.5
    assert mixer.alpha.grad is not None and mixer.alpha.grad.abs().sum() > 0


def test_per_graph_linear_selects_noncontiguous_rows():
    linear = PerGraphLinear(4, 1, 1).double()
    with torch.no_grad():
        linear.weight[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64, requires_grad=True)
    actual = linear(x, (3, 1))
    actual.sum().backward()
    torch.testing.assert_close(actual, torch.tensor([[[8.0]], [[6.0]]], dtype=torch.float64))
    torch.testing.assert_close(
        linear.weight.grad[:, 0, 0], torch.tensor([0.0, 3.0, 0.0, 2.0], dtype=torch.float64)
    )


def test_headwise_adapter_matches_full_gate_when_inputs_are_identical():
    torch.manual_seed(2)
    gate = ReferenceGate().double()
    adapter = _HeadwiseGateAdapter()
    hidden = torch.randn(4, 3, dtype=torch.float64)
    actual = torch.stack([adapter(gate, head, hidden, torch.zeros_like(hidden)) for head in range(2)])
    expected = gate(hidden.unsqueeze(0))[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_batched_adapter_matches_headwise_outputs_and_gradients():
    torch.manual_seed(9)
    headwise_gates = nn.ModuleList(
        [ReferenceGate(heads=2, groups=2).double() for _ in range(2)]
    )
    batched_gates = copy.deepcopy(headwise_gates)
    layer_ids = (0, 0, 1)
    head_ids = (0, 1, 0)
    headwise_hidden = torch.randn(3, 5, 3, dtype=torch.float64, requires_grad=True)
    batched_hidden = headwise_hidden.detach().clone().requires_grad_(True)
    headwise_delta = torch.randn(3, 5, 3, dtype=torch.float64, requires_grad=True)
    batched_delta = headwise_delta.detach().clone().requires_grad_(True)
    loss_weight = torch.randn(3, 5, dtype=torch.float64)
    adapter = _HeadwiseGateAdapter()

    expected = torch.stack(
        [
            adapter(
                headwise_gates[layer_id],
                head_id,
                headwise_hidden[index],
                headwise_delta[index],
            )
            for index, (layer_id, head_id) in enumerate(zip(layer_ids, head_ids))
        ]
    )
    actual = adapter.forward_batch(
        batched_gates,
        layer_ids,
        head_ids,
        batched_hidden,
        batched_delta,
    )
    (expected * loss_weight).sum().backward()
    (actual * loss_weight).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        batched_hidden.grad, headwise_hidden.grad, rtol=1e-8, atol=1e-8
    )
    torch.testing.assert_close(
        batched_delta.grad, headwise_delta.grad, rtol=1e-8, atol=1e-8
    )
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        headwise_gates.named_parameters(), batched_gates.named_parameters()
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=1e-8,
            atol=1e-8,
        )


def test_scorer_is_invariant_to_graph_microbatch_size_without_token_adjacency():
    torch.manual_seed(3)
    gates = [ReferenceGate() for _ in range(2)]
    scorer = ImplicitGraphScorer(
        gates,
        _config(),
        graph_dim=2,
        graph_microbatch_size=1,
        compute_dtype=torch.float64,
    ).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    one = scorer(x, microbatch_size=1, token_microbatch_size=2)
    many = scorer(x, microbatch_size=4, token_microbatch_size=3)
    torch.testing.assert_close(one, many, rtol=1e-12, atol=1e-12)


def test_implicit_path_never_forms_a_token_by_token_bmm_output(monkeypatch):
    mixer = ImplicitGraphMixer(1, 3, 2)
    outputs = []
    original = torch.bmm

    def record(left, right):
        result = original(left, right)
        outputs.append(result.shape)
        return result

    monkeypatch.setattr(torch, "bmm", record)
    mixer.prepare(torch.randn(1, 7, 3), (0,), token_microbatch_size=2)
    assert all(not (shape[-2] == 7 and shape[-1] == 7) for shape in outputs)


def test_graph_batch_and_microbatch_validation():
    batch = GraphBatch((0, 1), (0, 0), (0, 1))
    assert batch.graph_ids == (0, 1)
    assert resolve_graph_microbatch_size("auto", 3, 2) == 2
    assert resolve_graph_microbatch_size(6, 3, 2) == 6
