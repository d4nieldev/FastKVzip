"""Whole-context implicit graph scoring primitives.

The mixer implements the low-rank implicit adjacency from the experiment plan.
Production callers use the streamed helpers below, so no token-by-token
adjacency or full [graphs, tokens, hidden] mixer output is kept.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

ACTIVATION_ORDER = "batchnorm-leaky-relu"


def compute_dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPE_NAMES.items():
        if dtype == candidate:
            return name
    raise ValueError(f"unsupported compute dtype: {dtype}")


def parse_compute_dtype(value: object) -> torch.dtype:
    if not isinstance(value, str) or value not in _DTYPE_NAMES:
        raise ValueError(f"unsupported compute dtype: {value}")
    return _DTYPE_NAMES[value]


@dataclass(frozen=True)
class GraphBatch:
    """Python control metadata for one complete-graph microbatch."""

    graph_ids: tuple[int, ...]
    layer_ids: tuple[int, ...]
    head_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        identities = (self.graph_ids, self.layer_ids, self.head_ids)
        if any(
            type(values) is not tuple
            or not values
            or any(type(value) is not int for value in values)
            for values in identities
        ):
            raise TypeError(
                "graph batch identities must be non-empty tuples of Python integers"
            )
        if len({len(values) for values in identities}) != 1:
            raise ValueError("graph batch identities must have equal lengths")


def _graph_id_tuple(
    graph_ids: Sequence[int] | Tensor,
    *,
    num_graphs: int,
    expected_size: int | None = None,
) -> tuple[int, ...]:
    if isinstance(graph_ids, Tensor):
        if graph_ids.device.type != "cpu":
            raise ValueError("graph identity tensors must remain on CPU")
        if graph_ids.ndim != 1:
            raise ValueError("graph identity must be one-dimensional")
        graph_ids = tuple(graph_ids.tolist())
    else:
        graph_ids = tuple(graph_ids)
    if expected_size is not None and len(graph_ids) != expected_size:
        raise ValueError(f"expected {expected_size} graph IDs")
    if not graph_ids or any(
        type(graph_id) is not int or not 0 <= graph_id < num_graphs
        for graph_id in graph_ids
    ):
        raise ValueError("graph identity contains an unknown graph")
    return graph_ids


def _select_graph_rows(rows: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
    graph_ids = _graph_id_tuple(graph_ids, num_graphs=rows.size(0))
    start = graph_ids[0]
    if graph_ids == tuple(range(start, start + len(graph_ids))):
        return rows.narrow(0, start, len(graph_ids))
    index = torch.tensor(graph_ids, dtype=torch.long, device=rows.device)
    return rows.index_select(0, index)


def _reduction_dtype(*values: Tensor) -> torch.dtype:
    """Use FP32 in production and retain FP64 for gradient-equivalence tests."""

    return torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32


def resolve_graph_microbatch_size(value: str | int, layers: int, heads: int) -> int:
    """Resolve auto or validate a complete-graph microbatch size."""

    graph_count = layers * heads
    if value == "auto":
        return heads
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= graph_count:
        raise ValueError(f"graph microbatch size must be an integer from 1 to {graph_count}")
    return value


def subgraph_groups(
    token_count: int, subgraph_size: int, token_budget: int
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield equal-length subgraph starts that fit one token budget."""

    if token_count < 1 or subgraph_size < 1 or token_budget % subgraph_size:
        raise ValueError("subgraph sizes must be positive and divide the token microbatch size")
    full_count, tail = divmod(token_count, subgraph_size)
    per_group = token_budget // subgraph_size
    for first in range(0, full_count, per_group):
        yield tuple(
            index * subgraph_size
            for index in range(first, min(first + per_group, full_count))
        ), subgraph_size
    if tail:
        yield (full_count * subgraph_size,), tail


class PerGraphLinear(nn.Module):
    """Bias-free projections with independent rows for flattened layer/head graphs."""

    def __init__(
        self,
        num_graphs: int,
        in_features: int,
        out_features: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(num_graphs, out_features, in_features, device=device, dtype=dtype)
        )
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        weights = _select_graph_rows(self.weight, graph_ids).to(x.dtype)
        if x.ndim != 3 or x.size(0) != weights.size(0):
            raise ValueError("x and graph_ids must have shapes [B,T,D] and [B]")
        return torch.bmm(x, weights.transpose(1, 2))


@dataclass(frozen=True)
class ContextNormStats:
    """Current-context BatchNorm statistics (FP32 in production)."""

    mean: Tensor
    invstd: Tensor


@dataclass(frozen=True)
class PreparedImplicitGraph:
    """Compact state retained between streamed mixer passes."""

    graph_ids: tuple[int, ...]
    y1: Tensor
    gram: Tensor
    kernel: Tensor
    norm: ContextNormStats
    token_count: int


class ImplicitGraphMixer(nn.Module):
    """Per-graph low-rank mixer without materializing a token adjacency."""

    def __init__(
        self,
        num_graphs: int,
        hidden_dim: int,
        graph_dim: int,
        *,
        gram_normalization: str = "token-count",
        leaky_relu_slope: float = 0.01,
        alpha_init: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or hidden_dim < 1 or graph_dim < 1:
            raise ValueError("num_graphs, hidden_dim, and graph_dim must be positive")
        if gram_normalization not in {"token-count", "none"}:
            raise ValueError("gram_normalization must be token-count or none")
        if not math.isfinite(leaky_relu_slope) or leaky_relu_slope < 0:
            raise ValueError("leaky_relu_slope must be finite and non-negative")
        if not math.isfinite(alpha_init):
            raise ValueError("alpha_init must be finite")
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.gram_normalization = gram_normalization
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.in_proj = PerGraphLinear(
            num_graphs, hidden_dim, 2 * graph_dim, device=device, dtype=dtype
        )
        self.out_proj = PerGraphLinear(
            num_graphs, graph_dim, hidden_dim, device=device, dtype=dtype
        )
        self.gamma = nn.Parameter(torch.ones(num_graphs, hidden_dim, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(num_graphs, hidden_dim, device=device, dtype=dtype))
        self.alpha = nn.Parameter(torch.full((num_graphs,), alpha_init, device=device, dtype=dtype))

    @property
    def w1(self) -> Tensor:
        return self.in_proj.weight[:, : self.graph_dim]

    @property
    def w2(self) -> Tensor:
        return self.in_proj.weight[:, self.graph_dim :]

    @property
    def w(self) -> Tensor:
        return self.out_proj.weight

    @property
    def device(self) -> torch.device:
        return self.in_proj.weight.device

    def project(
        self, hidden: Tensor, graph_ids: Sequence[int] | Tensor
    ) -> tuple[Tensor, Tensor]:
        packed = self.in_proj(hidden, graph_ids)
        return packed.split(self.graph_dim, dim=-1)

    def _kernel(self, gram: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        weights = _select_graph_rows(self.out_proj.weight, graph_ids).to(gram.dtype)
        return torch.bmm(gram, weights.transpose(1, 2))

    def _raw(self, y1: Tensor, kernel: Tensor) -> Tensor:
        dtype = _reduction_dtype(y1, kernel)
        return torch.bmm(y1.to(dtype), kernel.to(dtype))

    @staticmethod
    def _chunks(token_count: int, token_microbatch_size: int):
        for start in range(0, token_count, token_microbatch_size):
            yield start, min(start + token_microbatch_size, token_count)

    @staticmethod
    def _context_norm_stats(raw_chunks: Iterator[Tensor]) -> ContextNormStats:
        count = 0
        mean = None
        m2 = None
        for raw in raw_chunks:
            values = raw.to(_reduction_dtype(raw))
            chunk_count = values.size(1)
            chunk_mean = values.mean(dim=1)
            chunk_m2 = (values - chunk_mean.unsqueeze(1)).square().sum(dim=1)
            if mean is None:
                count = chunk_count
                mean = chunk_mean
                m2 = chunk_m2
                continue
            total = count + chunk_count
            delta = chunk_mean - mean
            mean = mean + delta * (chunk_count / total)
            m2 = m2 + chunk_m2 + delta.square() * (count * chunk_count / total)
            count = total
        if mean is None or m2 is None or count < 1:
            raise ValueError("context normalization requires at least one token")
        variance = m2 / count
        return ContextNormStats(mean=mean, invstd=torch.rsqrt(variance + 1e-5))

    def prepare_from_chunks(
        self,
        chunks: Iterator[tuple[int, Tensor]],
        *,
        graph_ids: Sequence[int] | Tensor,
        token_count: int,
        token_microbatch_size: int,
    ) -> PreparedImplicitGraph:
        """Project once, retain Y1, and stream Gram and context statistics."""

        graph_ids = _graph_id_tuple(graph_ids, num_graphs=self.num_graphs)
        if token_count < 1 or token_microbatch_size < 1:
            raise ValueError("token_count and token_microbatch_size must be positive")
        y1 = None
        gram = None
        expected_start = 0
        for start, hidden in chunks:
            stop = start + hidden.size(1)
            if start != expected_start or stop > token_count:
                raise ValueError("mixer chunks must cover the context in order")
            hidden = hidden.to(device=self.device)
            first, second = self.project(hidden, graph_ids)
            if y1 is None:
                y1 = torch.empty(
                    (len(graph_ids), token_count, self.graph_dim),
                    device=self.device,
                    dtype=first.dtype,
                )
                gram = torch.zeros(
                    (len(graph_ids), self.graph_dim, self.graph_dim),
                    device=self.device,
                    dtype=_reduction_dtype(first, second),
                )
            y1[:, start:stop] = first
            gram += torch.bmm(
                first.to(gram.dtype).transpose(1, 2), second.to(gram.dtype)
            )
            expected_start = stop
        if y1 is None or gram is None or expected_start != token_count:
            raise ValueError("mixer chunks do not cover the complete context")
        if self.gram_normalization == "token-count":
            gram = gram / token_count
        kernel = self._kernel(gram, graph_ids)
        norm = self._context_norm_stats(
            self._raw(y1[:, start:stop], kernel)
            for start, stop in self._chunks(token_count, token_microbatch_size)
        )
        return PreparedImplicitGraph(graph_ids, y1, gram, kernel, norm, token_count)

    def prepare(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        token_microbatch_size: int,
    ) -> PreparedImplicitGraph:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        token_count = hidden.size(1)

        def chunks():
            for start, stop in self._chunks(token_count, token_microbatch_size):
                yield start, hidden[:, start:stop]

        return self.prepare_from_chunks(
            chunks(),
            graph_ids=graph_ids,
            token_count=token_count,
            token_microbatch_size=token_microbatch_size,
        )

    def normalized(self, raw: Tensor, prepared: PreparedImplicitGraph) -> Tensor:
        return (raw - prepared.norm.mean.unsqueeze(1)) * prepared.norm.invstd.unsqueeze(1)

    def activated(
        self, normalized: Tensor, graph_ids: Sequence[int] | Tensor
    ) -> Tensor:
        gamma = _select_graph_rows(self.gamma, graph_ids).to(normalized.dtype).unsqueeze(1)
        beta = _select_graph_rows(self.beta, graph_ids).to(normalized.dtype).unsqueeze(1)
        return F.leaky_relu(
            gamma * normalized + beta,
            negative_slope=self.leaky_relu_slope,
        )

    def delta(
        self,
        y1: Tensor,
        prepared: PreparedImplicitGraph,
        graph_ids: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        ids = prepared.graph_ids if graph_ids is None else graph_ids
        raw = self._raw(y1, prepared.kernel)
        normalized = self.normalized(raw, prepared)
        alpha = _select_graph_rows(self.alpha, ids).to(raw.dtype).view(-1, 1, 1)
        return alpha * self.activated(normalized, ids)

    def forward(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        """Convenience full-output path for small tests; scoring uses streamed APIs."""

        prepared = self.prepare(
            hidden, graph_ids, token_microbatch_size=max(1, hidden.size(1))
        )
        return self.delta(prepared.y1, prepared)


class _HeadwiseGateAdapter(nn.Module):
    """Apply a head-specific hidden-state delta to matching gate slices."""

    def forward(self, gate: nn.Module, head: int, hidden: Tensor, delta: Tensor) -> Tensor:
        token_count = hidden.size(0)
        gate_dim = gate.output_dim
        groups = gate.ngroup
        mixed = hidden + delta

        q_weight = gate.q_proj.weight.view(
            gate.nhead, groups * gate_dim, gate.q_proj.in_features
        )[head].to(mixed.dtype)
        q_bias = None
        if gate.q_proj.bias is not None:
            q_bias = gate.q_proj.bias.view(gate.nhead, groups * gate_dim)[head].to(
                mixed.dtype
            )
        queries = F.linear(mixed, q_weight, q_bias)
        queries = gate.q_norm(queries.view(token_count, groups, gate_dim))

        k_weight = gate.k_proj.weight.view(
            gate.nhead, gate_dim, gate.k_proj.in_features
        )[head].to(mixed.dtype)
        keys = gate.k_norm(F.linear(mixed, k_weight))

        logits = torch.einsum("tr,tgr->tg", keys, queries) / gate.d
        logits = logits + gate.b[head, 0].to(mixed.dtype)
        base_logits = torch.einsum(
            "sr,tgr->tsg", gate.k_base[head, 0].to(queries.dtype), queries
        ) / gate.d
        scores = 1 / (1 + torch.exp(base_logits - logits.unsqueeze(1)).sum(dim=1))
        return scores.mean(dim=-1)

    def forward_batch(
        self,
        gates: Sequence[nn.Module],
        layer_ids: Sequence[int],
        head_ids: Sequence[int],
        hidden: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Apply matching gate heads to a complete graph microbatch."""

        layer_ids = tuple(layer_ids)
        head_ids = tuple(head_ids)
        graph_count = len(layer_ids)
        if (
            not graph_count
            or hidden.ndim != 3
            or delta.shape != hidden.shape
            or len(head_ids) != graph_count
            or hidden.size(0) != graph_count
        ):
            raise ValueError(
                "hidden and delta must match [graphs,tokens,hidden_dim] identities"
            )

        selected = tuple(
            (gates[layer_id], head_id)
            for layer_id, head_id in zip(layer_ids, head_ids)
        )
        mixed = hidden + delta
        token_count = mixed.size(1)
        first_gate = selected[0][0]
        gate_dim = first_gate.output_dim
        groups = first_gate.ngroup

        q_weight = torch.stack(
            [
                gate.q_proj.weight.reshape(
                    gate.nhead, groups * gate_dim, gate.q_proj.in_features
                )[head]
                for gate, head in selected
            ]
        ).to(mixed.dtype)
        queries = torch.bmm(mixed, q_weight.transpose(1, 2))
        queries = queries + torch.stack(
            [
                gate.q_proj.bias.reshape(gate.nhead, groups * gate_dim)[head]
                for gate, head in selected
            ]
        ).to(mixed.dtype).unsqueeze(1)
        queries = queries.reshape(graph_count, token_count, groups, gate_dim)

        k_weight = torch.stack(
            [
                gate.k_proj.weight.reshape(
                    gate.nhead, gate_dim, gate.k_proj.in_features
                )[head]
                for gate, head in selected
            ]
        ).to(mixed.dtype)
        keys = torch.bmm(mixed, k_weight.transpose(1, 2))

        def normalize(values: Tensor, attribute: str) -> Tensor:
            result = values
            for layer_id in dict.fromkeys(layer_ids):
                positions = torch.tensor(
                    [
                        index
                        for index, candidate in enumerate(layer_ids)
                        if candidate == layer_id
                    ],
                    device=mixed.device,
                )
                normalized = getattr(gates[layer_id], attribute)(
                    values.index_select(0, positions)
                )
                result = result.index_copy(0, positions, normalized)
            return result

        queries = normalize(queries, "q_norm")
        keys = normalize(keys, "k_norm")

        logits = torch.einsum("mtr,mtgr->mtg", keys, queries) / first_gate.d
        logits = logits + torch.stack(
            [gate.b[head, 0] for gate, head in selected]
        ).to(mixed.dtype).unsqueeze(1)
        k_base = torch.stack(
            [gate.k_base[head, 0] for gate, head in selected]
        ).to(queries.dtype)
        base_logits = torch.einsum("msr,mtgr->mtsg", k_base, queries) / first_gate.d
        scores = 1 / (
            1 + torch.exp(base_logits - logits.unsqueeze(2)).sum(dim=2)
        )
        return scores.mean(dim=-1)


class ImplicitGraphScorer(nn.Module):
    """Score one context with independent implicit mixers per layer and KV head."""

    def __init__(
        self,
        gates,
        model_config,
        *,
        graph_dim: int = 32,
        graph_microbatch_size: str | int = "auto",
        gram_normalization: str = "token-count",
        leaky_relu_slope: float = 0.01,
        alpha_init: float = 0.1,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config = getattr(model_config, "text_config", model_config)
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_key_value_heads
        self.num_graphs = self.num_layers * self.num_heads
        if len(gates) != self.num_layers:
            raise ValueError("one runtime gate is required per model layer")
        self.gates = nn.ModuleList(gates)
        first_gate = self.gates[0]
        original_compute_dtype = first_gate.q_proj.weight.dtype
        self.compute_dtype = original_compute_dtype if compute_dtype is None else compute_dtype
        compute_dtype_name(self.compute_dtype)
        if any(gate.q_proj.weight.dtype != original_compute_dtype for gate in self.gates):
            raise ValueError("all runtime gates must use the same compute dtype")
        device = first_gate.q_proj.weight.device
        master_dtype = (
            torch.float32
            if self.compute_dtype in {torch.float16, torch.bfloat16}
            else self.compute_dtype
        )
        self.gates.to(device=device, dtype=master_dtype)
        self.hidden_dim = first_gate.q_proj.in_features
        self.gate_dim = first_gate.output_dim
        if any(
            gate.nhead != self.num_heads
            or gate.q_proj.in_features != self.hidden_dim
            or gate.output_dim != self.gate_dim
            for gate in self.gates
        ):
            raise ValueError("runtime gate dimensions do not match model configuration")
        resolve_graph_microbatch_size(graph_microbatch_size, self.num_layers, self.num_heads)
        self.graph_microbatch_size = graph_microbatch_size
        self.mixer = ImplicitGraphMixer(
            self.num_graphs,
            self.hidden_dim,
            graph_dim,
            gram_normalization=gram_normalization,
            leaky_relu_slope=leaky_relu_slope,
            alpha_init=alpha_init,
            device=device,
            dtype=master_dtype,
        )
        self._gate_adapter = _HeadwiseGateAdapter()

    @property
    def device(self) -> torch.device:
        return self.mixer.device

    @property
    def graph_dim(self) -> int:
        return self.mixer.graph_dim

    def graph_batches(
        self, *, microbatch_size: str | int | None = None
    ) -> Iterator[GraphBatch]:
        size = resolve_graph_microbatch_size(
            self.graph_microbatch_size if microbatch_size is None else microbatch_size,
            self.num_layers,
            self.num_heads,
        )
        for start in range(0, self.num_graphs, size):
            graph_ids = tuple(range(start, min(start + size, self.num_graphs)))
            yield GraphBatch(
                graph_ids,
                tuple(graph_id // self.num_heads for graph_id in graph_ids),
                tuple(graph_id % self.num_heads for graph_id in graph_ids),
            )

    def prepare(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        token_microbatch_size: int,
    ) -> PreparedImplicitGraph:
        hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
        return self.mixer.prepare(
            hidden, graph_ids, token_microbatch_size=token_microbatch_size
        )

    def score_prepared(
        self,
        hidden: Tensor,
        prepared: PreparedImplicitGraph,
        *,
        layer_ids: Sequence[int] | Tensor,
        head_ids: Sequence[int] | Tensor,
    ) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or hidden.size(0) != len(prepared.graph_ids):
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        layer_ids = _graph_id_tuple(
            layer_ids, num_graphs=self.num_layers, expected_size=hidden.size(0)
        )
        head_ids = _graph_id_tuple(
            head_ids, num_graphs=self.num_heads, expected_size=hidden.size(0)
        )
        delta = self.mixer.delta(prepared.y1, prepared)
        scores = self._gate_adapter.forward_batch(
            self.gates,
            layer_ids,
            head_ids,
            hidden.to(device=self.device, dtype=self.compute_dtype),
            delta,
        )
        return scores, delta

    def forward(
        self,
        hidden: Tensor,
        *,
        microbatch_size: str | int | None = None,
        token_microbatch_size: int = 1000,
    ) -> Tensor:
        if hidden.ndim == 4 and hidden.size(1) == 1:
            hidden = hidden[:, 0]
        if hidden.ndim != 3 or hidden.size(0) != self.num_layers:
            raise ValueError("hidden must have shape [layers,T,D] or [layers,1,T,D]")
        if hidden.size(-1) != self.hidden_dim:
            raise ValueError(f"expected hidden dimension {self.hidden_dim}")
        token_count = hidden.size(1)
        score_batches = []
        for batch in self.graph_batches(microbatch_size=microbatch_size):
            def chunks():
                for start in range(0, token_count, token_microbatch_size):
                    stop = min(start + token_microbatch_size, token_count)
                    yield start, torch.stack(
                        tuple(hidden[layer_id, start:stop] for layer_id in batch.layer_ids)
                    ).to(device=self.device, dtype=self.compute_dtype)

            prepared = self.mixer.prepare_from_chunks(
                chunks(),
                graph_ids=batch.graph_ids,
                token_count=token_count,
                token_microbatch_size=token_microbatch_size,
            )
            chunks_scores = []
            for start in range(0, token_count, token_microbatch_size):
                stop = min(start + token_microbatch_size, token_count)
                graph_hidden = torch.stack(
                    tuple(hidden[layer_id, start:stop] for layer_id in batch.layer_ids)
                ).to(device=self.device, dtype=self.compute_dtype)
                slice_prepared = PreparedImplicitGraph(
                    batch.graph_ids,
                    prepared.y1[:, start:stop],
                    prepared.gram,
                    prepared.kernel,
                    prepared.norm,
                    token_count,
                )
                scores, _ = self.score_prepared(
                    graph_hidden,
                    slice_prepared,
                    layer_ids=batch.layer_ids,
                    head_ids=batch.head_ids,
                )
                chunks_scores.append(scores)
            score_batches.append(torch.cat(chunks_scores, dim=1))
        flat_scores = torch.cat(score_batches, dim=0)
        return flat_scores.view(self.num_layers, self.num_heads, token_count).unsqueeze(1)
