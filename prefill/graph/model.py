"""Whole-context implicit graph scoring primitives.

The mixer implements the low-rank implicit adjacency from the experiment plan.
Production callers use the streamed helpers below, so no token-by-token
adjacency or full [graphs, tokens, hidden] mixer output is kept.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
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

ACTIVATION_ORDER = "normalization-leaky-relu"
LEGACY_ACTIVATION_ORDER = "batchnorm-leaky-relu"

NORMALIZATIONS = ("none", "batchnorm", "granola")
NORMALIZATION_SHARING = ("graph", "layer", "global")
GRANOLA_ADAPTIVITY = ("graph", "token")

# Every normalization setting a checkpoint records. The first names the mode;
# a checkpoint that has it must have all of them.
NORMALIZATION_CONFIG_KEYS = (
    "normalization",
    "normalization_sharing",
    "granola_gnn_depth",
    "granola_mlp_depth",
    "granola_rnf_dim",
    "granola_adaptivity",
    "normalization_seed",
)

# Stand-in RNF width for a checkpoint that records no graph dimension. Nothing
# reads it without a mixer, but it still has to be a usable width.
DEFAULT_GRANOLA_RNF_DIM = 32


def canonical_normalization_config(config: Mapping[str, object]) -> dict[str, object]:
    """Fill in the normalization settings a pre-normalization checkpoint lacks.

    Checkpoints written before this option existed carry no normalization keys
    at all. They described the BatchNorm mixer, so they get its settings. A
    gate-only checkpoint records no graph dimension, so the GraNoLa width falls
    back to a fixed default rather than to nothing.

    Both training entry points and the evaluator share this, so a checkpoint
    means the same thing to all of them.
    """

    canonical = copy.deepcopy(dict(config))
    if "normalization" in canonical:
        missing = [
            name for name in NORMALIZATION_CONFIG_KEYS if name not in canonical
        ]
        if missing:
            raise ValueError(
                f"checkpoint normalization config is missing: {', '.join(missing)}"
            )
        return canonical
    graph_dim = canonical.get("graph_dim")
    canonical.update(
        {
            "normalization": "batchnorm",
            "normalization_sharing": "graph",
            "granola_gnn_depth": 1,
            "granola_mlp_depth": 1,
            "granola_rnf_dim": (
                DEFAULT_GRANOLA_RNF_DIM if graph_dim is None else graph_dim
            ),
            "granola_adaptivity": "graph",
            "normalization_seed": 0,
        }
    )
    if canonical.get("activation_order") == LEGACY_ACTIVATION_ORDER:
        canonical["activation_order"] = ACTIVATION_ORDER
    return canonical


def compute_dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPE_NAMES.items():
        if dtype == candidate:
            return name
    raise ValueError(f"unsupported compute dtype: {dtype}")


def parse_compute_dtype(value: object) -> torch.dtype:
    if not isinstance(value, str) or value not in _DTYPE_NAMES:
        raise ValueError(f"unsupported compute dtype: {value}")
    return _DTYPE_NAMES[value]


def derive_evaluation_rnf_seed(
    base_seed: int, dataset_name: str, dataset_index: int
) -> int:
    """Derive a stable per-example RNF seed from the recorded training seed."""

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed < 2**63
        or not isinstance(dataset_name, str)
        or not dataset_name
        or isinstance(dataset_index, bool)
        or not isinstance(dataset_index, int)
        or dataset_index < 0
    ):
        raise ValueError("evaluation RNF seed requires a valid seed and dataset identity")
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(base_seed).encode())
    digest.update(b"\0")
    digest.update(dataset_name.encode())
    digest.update(b"\0")
    digest.update(str(dataset_index).encode())
    return int.from_bytes(digest.digest(), "little") % (2**63 - 1)


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
    """Indexed projections with independent parameter rows."""

    def __init__(
        self,
        num_graphs: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(num_graphs, out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_graphs, out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        weights = _select_graph_rows(self.weight, graph_ids).to(x.dtype)
        if x.ndim != 3 or x.size(0) != weights.size(0):
            raise ValueError("x and graph_ids must have shapes [B,T,D] and [B]")
        output = torch.bmm(x, weights.transpose(1, 2))
        if self.bias is not None:
            bias = _select_graph_rows(self.bias, graph_ids).to(x.dtype).unsqueeze(1)
            output = output + bias
        return output


class _PerGroupLayerNorm(nn.Module):
    """Last-dimension LayerNorm with affine rows selected by group identity."""

    def __init__(
        self, num_groups: int, features: int, *, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_groups, features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_groups, features, device=device, dtype=dtype)
        )

    def forward(self, values: Tensor, group_ids: Sequence[int] | Tensor) -> Tensor:
        dtype = _reduction_dtype(values)
        reduced = values.to(dtype)
        mean = reduced.mean(dim=-1, keepdim=True)
        variance = (reduced - mean).square().mean(dim=-1, keepdim=True)
        normalized = (reduced - mean) * torch.rsqrt(variance + 1e-5)
        weight = _select_graph_rows(self.weight, group_ids).to(dtype).unsqueeze(1)
        bias = _select_graph_rows(self.bias, group_ids).to(dtype).unsqueeze(1)
        return weight * normalized + bias


class _PerGroupMLP(nn.Module):
    """DEAR-shaped indexed MLP: Linear, then (LayerNorm, ReLU, Linear)."""

    def __init__(
        self,
        num_groups: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int,
        *,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be positive")
        self.linears = nn.ModuleList(
            [
                PerGraphLinear(
                    num_groups,
                    in_features,
                    out_features if depth == 1 else hidden_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            ]
        )
        self.norms = nn.ModuleList()
        for index in range(1, depth):
            self.norms.append(
                _PerGroupLayerNorm(
                    num_groups, hidden_features, device=device, dtype=dtype
                )
            )
            self.linears.append(
                PerGraphLinear(
                    num_groups,
                    hidden_features,
                    out_features if index == depth - 1 else hidden_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )

    def first(self, values: Tensor, group_ids: Sequence[int] | Tensor) -> Tensor:
        return self.linears[0](values, group_ids)

    def finish(self, values: Tensor, group_ids: Sequence[int] | Tensor) -> Tensor:
        for norm, linear in zip(self.norms, self.linears[1:]):
            values = linear(F.relu(norm(values, group_ids)), group_ids)
        return values

    def forward(self, values: Tensor, group_ids: Sequence[int] | Tensor) -> Tensor:
        return self.finish(self.first(values, group_ids), group_ids)


@dataclass(frozen=True)
class ContextNormStats:
    """Current-context BatchNorm statistics (FP32 in production)."""

    mean: Tensor
    invstd: Tensor


@dataclass(frozen=True)
class _GranolaNormState:
    """GraNoLa state retained between streamed mixer passes.

    `rnf` and `token_hidden` are token-major, so a token slice selects from
    them. `pooled` and `stats` describe the whole context, so slicing must
    leave them alone; recomputing either from one token chunk would make the
    scores depend on the microbatch split.
    """

    rnf: Tensor
    token_hidden: Tensor | None = None
    pooled: Tensor | None = None
    stats: ContextNormStats | None = None

    def readout(self) -> Tensor:
        """The GNN output the affine heads read: one row per graph or per token."""

        source = self.pooled if self.pooled is not None else self.token_hidden
        if source is None:
            raise ValueError("prepared graph is missing the GraNoLa readout")
        return source


@dataclass(frozen=True)
class PreparedImplicitGraph:
    """Compact state retained between streamed mixer passes."""

    graph_ids: tuple[int, ...]
    y1: Tensor
    # The GraNoLa GNN reads the message features directly, so only that branch
    # retains them; BatchNorm folds them into the Gram matrix and drops them.
    y2: Tensor | None
    gram: Tensor
    # The Gram matrix folded with the out projection, so the other branches
    # reach hidden width in one product. GraNoLa applies the out projection
    # after its affine instead, so folding it in would reorder the model and
    # the branch leaves this empty.
    kernel: Tensor | None
    norm: ContextNormStats | _GranolaNormState | None
    token_count: int

    def select_tokens(self, positions: Tensor) -> PreparedImplicitGraph:
        index = positions.to(self.y1.device)
        norm = self.norm
        if isinstance(norm, _GranolaNormState):
            norm = _GranolaNormState(
                norm.rnf.index_select(1, index),
                None
                if norm.token_hidden is None
                else norm.token_hidden.index_select(1, index),
                norm.pooled,
                norm.stats,
            )
        return PreparedImplicitGraph(
            self.graph_ids,
            self.y1.index_select(1, index),
            None if self.y2 is None else self.y2.index_select(1, index),
            self.gram,
            self.kernel,
            norm,
            self.token_count,
        )

    def detached_to(self, device: str | torch.device) -> PreparedImplicitGraph:
        norm = self.norm
        if isinstance(norm, ContextNormStats):
            norm = ContextNormStats(
                norm.mean.detach().to(device), norm.invstd.detach().to(device)
            )
        elif isinstance(norm, _GranolaNormState):
            stats = norm.stats
            norm = _GranolaNormState(
                norm.rnf.detach().to(device),
                None
                if norm.token_hidden is None
                else norm.token_hidden.detach().to(device),
                None if norm.pooled is None else norm.pooled.detach().to(device),
                None
                if stats is None
                else ContextNormStats(
                    stats.mean.detach().to(device), stats.invstd.detach().to(device)
                ),
            )
        return PreparedImplicitGraph(
            self.graph_ids,
            self.y1.detach().to(device),
            None if self.y2 is None else self.y2.detach().to(device),
            self.gram.detach().to(device),
            None if self.kernel is None else self.kernel.detach().to(device),
            norm,
            self.token_count,
        )


class ImplicitGraphMixer(nn.Module):
    """Per-graph low-rank mixer without materializing a token adjacency."""

    def __init__(
        self,
        num_graphs: int,
        hidden_dim: int,
        graph_dim: int,
        *,
        num_heads: int = 1,
        normalization: str = "batchnorm",
        normalization_sharing: str = "graph",
        granola_gnn_depth: int = 1,
        granola_mlp_depth: int = 1,
        granola_rnf_dim: int | None = None,
        granola_adaptivity: str = "graph",
        normalization_seed: int = 0,
        gram_normalization: str = "token-count",
        leaky_relu_slope: float = 0.01,
        alpha_init: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or hidden_dim < 1 or graph_dim < 1 or num_heads < 1:
            raise ValueError(
                "num_graphs, hidden_dim, graph_dim, and num_heads must be positive"
            )
        if num_graphs % num_heads:
            raise ValueError("num_graphs must be divisible by num_heads")
        if normalization not in NORMALIZATIONS:
            raise ValueError("normalization must be none, batchnorm, or granola")
        if normalization_sharing not in NORMALIZATION_SHARING:
            raise ValueError("normalization sharing must be graph, layer, or global")
        if (
            isinstance(granola_gnn_depth, bool)
            or not isinstance(granola_gnn_depth, int)
            or granola_gnn_depth < 1
        ):
            raise ValueError("GraNoLa GNN depth must be a positive integer")
        if (
            isinstance(granola_mlp_depth, bool)
            or not isinstance(granola_mlp_depth, int)
            or granola_mlp_depth < 1
        ):
            raise ValueError("GraNoLa MLP depth must be a positive integer")
        if granola_rnf_dim is None:
            granola_rnf_dim = graph_dim
        if (
            isinstance(granola_rnf_dim, bool)
            or not isinstance(granola_rnf_dim, int)
            or granola_rnf_dim < 1
        ):
            raise ValueError("GraNoLa RNF dimension must be a positive integer")
        if granola_adaptivity not in GRANOLA_ADAPTIVITY:
            raise ValueError("GraNoLa adaptivity must be graph or token")
        if (
            isinstance(normalization_seed, bool)
            or not isinstance(normalization_seed, int)
            or not 0 <= normalization_seed < 2**63
        ):
            raise ValueError("normalization seed must be an integer from 0 to 2^63-1")
        if gram_normalization not in {"token-count", "none"}:
            raise ValueError("gram_normalization must be token-count or none")
        if not math.isfinite(leaky_relu_slope) or leaky_relu_slope < 0:
            raise ValueError("leaky_relu_slope must be finite and non-negative")
        if not math.isfinite(alpha_init):
            raise ValueError("alpha_init must be finite")
        self.num_graphs = num_graphs
        self.num_heads = num_heads
        self.num_layers = num_graphs // num_heads
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.normalization = normalization
        self.normalization_sharing = normalization_sharing
        self.granola_gnn_depth = granola_gnn_depth
        self.granola_mlp_depth = granola_mlp_depth
        self.granola_rnf_dim = granola_rnf_dim
        self.granola_adaptivity = granola_adaptivity
        self.normalization_seed = normalization_seed
        self.num_normalization_groups = {
            "graph": num_graphs,
            "layer": self.num_layers,
            "global": 1,
        }[normalization_sharing]
        self.gram_normalization = gram_normalization
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.in_proj = PerGraphLinear(
            num_graphs, hidden_dim, 2 * graph_dim, device=device, dtype=dtype
        )
        self.out_proj = PerGraphLinear(
            num_graphs, graph_dim, hidden_dim, device=device, dtype=dtype
        )
        if normalization == "batchnorm":
            self.gamma = nn.Parameter(
                torch.ones(
                    self.num_normalization_groups,
                    hidden_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.beta = nn.Parameter(
                torch.zeros(
                    self.num_normalization_groups,
                    hidden_dim,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.alpha = nn.Parameter(
            torch.full((num_graphs,), alpha_init, device=device, dtype=dtype)
        )
        self.granola_blocks = nn.ModuleList()
        self.granola_gamma_head = None
        self.granola_beta_head = None
        if normalization == "granola":
            for index in range(granola_gnn_depth):
                self.granola_blocks.append(
                    _PerGroupMLP(
                        self.num_normalization_groups,
                        graph_dim + granola_rnf_dim if index == 0 else graph_dim,
                        graph_dim,
                        graph_dim,
                        granola_mlp_depth,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    )
                )
            self.granola_gamma_head = _PerGroupMLP(
                self.num_normalization_groups,
                graph_dim,
                graph_dim,
                graph_dim,
                2,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.granola_beta_head = _PerGroupMLP(
                self.num_normalization_groups,
                graph_dim,
                graph_dim,
                graph_dim,
                2,
                bias=True,
                device=device,
                dtype=dtype,
            )

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

    def normalization_group_ids(
        self, graph_ids: Sequence[int] | Tensor
    ) -> tuple[int, ...]:
        graph_ids = _graph_id_tuple(graph_ids, num_graphs=self.num_graphs)
        if self.normalization_sharing == "graph":
            return graph_ids
        if self.normalization_sharing == "layer":
            return tuple(graph_id // self.num_heads for graph_id in graph_ids)
        return (0,) * len(graph_ids)

    def next_rnf_seed(self) -> int:
        return int(torch.randint(0, 2**63 - 1, (), device="cpu").item())

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

    @staticmethod
    def _node_layer_norm(values: Tensor) -> Tensor:
        dtype = _reduction_dtype(values)
        reduced = values.to(dtype)
        mean = reduced.mean(dim=-1, keepdim=True)
        variance = (reduced - mean).square().mean(dim=-1, keepdim=True)
        return (reduced - mean) * torch.rsqrt(variance + 1e-5)

    def _sample_rnf(
        self,
        graph_ids: tuple[int, ...],
        token_count: int,
        *,
        seed: int,
        dtype: torch.dtype,
        offsets: Sequence[int] | None = None,
    ) -> Tensor:
        modulus = 2**63
        samples = []
        for row, graph_id in enumerate(graph_ids):
            graph_seed = seed + 0x1E3779B97F4A7C1 * (graph_id + 1)
            if offsets is not None:
                # Stacked subgraphs repeat a graph id, so the row's context
                # offset joins the key. Without it every subgraph of one
                # layer/head would receive the same random node features.
                graph_seed += 0x9E3779B97F4A7C15 * (int(offsets[row]) + 1)
            graph_seed %= modulus
            generator = torch.Generator(device=self.device)
            generator.manual_seed(graph_seed)
            samples.append(
                torch.randn(
                    (token_count, self.granola_rnf_dim),
                    device=self.device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        return torch.stack(samples)

    def messages(self, y1: Tensor, gram: Tensor) -> Tensor:
        """Aggregated messages A R, at graph width, before the out projection."""

        dtype = _reduction_dtype(y1, gram)
        return torch.bmm(y1.to(dtype), gram.to(dtype))

    def granola_gnn(
        self,
        y1: Tensor,
        y2: Tensor,
        rnf: Tensor,
        group_ids: Sequence[int],
        *,
        scale: int,
    ) -> Tensor:
        """Propagate the message features and the RNF over the implicit graph.

        Each block is a signed weighted GIN update, `Q = P + A P`, evaluated as
        `Y1 (Y1^T P) / scale` so the token-by-token adjacency is never formed.
        Everything here is graph width, so no token chunking is needed.
        """

        dtype = _reduction_dtype(y1, y2)
        y1 = y1.to(dtype)
        values = torch.cat((y2.to(dtype), rnf.to(dtype)), dim=-1)
        for block in self.granola_blocks:
            projected = block.first(values, group_ids).to(dtype)
            contraction = torch.bmm(y1.transpose(1, 2), projected) / scale
            values = block.finish(
                projected + torch.bmm(y1, contraction), group_ids
            )
        return values

    def granola_readout(self, hidden: Tensor) -> Tensor:
        """The GNN output the affine heads read.

        `graph` adaptivity pools over tokens to one row per graph; `token`
        adaptivity keeps one row per token.
        """

        if self.granola_adaptivity == "graph":
            return hidden.mean(dim=1, keepdim=True)
        return hidden

    def granola_normalized(self, messages: Tensor) -> Tensor:
        """Normalize aggregated messages from their own statistics.

        Scoring reuses the statistics stored when the graph was prepared; this
        recomputes them so training can differentiate through them. Both go
        through the same definition so the two cannot drift apart.
        """

        if self.granola_adaptivity == "token":
            return self._node_layer_norm(messages)
        stats = self._context_norm_stats(iter((messages,)))
        return (messages - stats.mean.unsqueeze(1)) * stats.invstd.unsqueeze(1)

    def _prepare_granola(
        self,
        y1: Tensor,
        y2: Tensor,
        gram: Tensor,
        graph_ids: tuple[int, ...],
        *,
        token_count: int,
        offsets: Sequence[int] | None,
        rnf_seed: int,
    ) -> _GranolaNormState:
        dtype = _reduction_dtype(y1, y2)
        rnf = self._sample_rnf(
            graph_ids, token_count, seed=rnf_seed, dtype=dtype, offsets=offsets
        )
        group_ids = self.normalization_group_ids(graph_ids)
        scale = token_count if self.gram_normalization == "token-count" else 1
        hidden = self.granola_gnn(y1, y2, rnf, group_ids, scale=scale)
        if self.granola_adaptivity == "token":
            return _GranolaNormState(rnf, token_hidden=hidden)
        # One affine pair per graph. The readout and the statistics both cover
        # the whole context, so they are computed once here and never from a
        # token chunk, which would make scores depend on the microbatch split.
        return _GranolaNormState(
            rnf,
            pooled=self.granola_readout(hidden),
            stats=self._context_norm_stats(iter((self.messages(y1, gram),))),
        )

    def prepare_from_chunks(
        self,
        chunks: Iterator[tuple[int, Tensor]],
        *,
        graph_ids: Sequence[int] | Tensor,
        token_count: int,
        token_microbatch_size: int,
        rnf_seed: int | None = None,
        offsets: Sequence[int] | None = None,
    ) -> PreparedImplicitGraph:
        """Project once, retain Y1, and stream Gram and context statistics."""

        graph_ids = _graph_id_tuple(graph_ids, num_graphs=self.num_graphs)
        if token_count < 1 or token_microbatch_size < 1:
            raise ValueError("token_count and token_microbatch_size must be positive")
        y1 = None
        y2 = None
        gram = None
        keep_y2 = self.normalization == "granola"
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
                if keep_y2:
                    y2 = torch.empty_like(y1)
            y1[:, start:stop] = first
            if y2 is not None:
                y2[:, start:stop] = second
            gram += torch.bmm(
                first.to(gram.dtype).transpose(1, 2), second.to(gram.dtype)
            )
            expected_start = stop
        if y1 is None or gram is None or expected_start != token_count:
            raise ValueError("mixer chunks do not cover the complete context")
        if self.gram_normalization == "token-count":
            gram = gram / token_count
        # Structurally inapplicable to GraNoLa, not merely unused: its affine
        # runs at graph width, before the out projection this folds in.
        kernel = None if self.normalization == "granola" else self._kernel(gram, graph_ids)
        if self.normalization == "batchnorm":
            norm: ContextNormStats | _GranolaNormState | None = self._context_norm_stats(
                self._raw(y1[:, start:stop], kernel)
                for start, stop in self._chunks(token_count, token_microbatch_size)
            )
        elif self.normalization == "granola":
            if rnf_seed is None:
                rnf_seed = self.next_rnf_seed() if self.training else self.normalization_seed
            if (
                isinstance(rnf_seed, bool)
                or not isinstance(rnf_seed, int)
                or not 0 <= rnf_seed < 2**63
            ):
                raise ValueError("RNF seed must be an integer from 0 to 2^63-1")
            assert y2 is not None
            norm = self._prepare_granola(
                y1,
                y2,
                gram,
                graph_ids,
                token_count=token_count,
                offsets=offsets,
                rnf_seed=rnf_seed,
            )
        else:
            norm = None
        return PreparedImplicitGraph(
            graph_ids, y1, y2, gram, kernel, norm, token_count
        )

    def prepare(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        token_microbatch_size: int,
        rnf_seed: int | None = None,
        offsets: Sequence[int] | None = None,
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
            rnf_seed=rnf_seed,
            offsets=offsets,
        )

    def normalized(self, values: Tensor, prepared: PreparedImplicitGraph) -> Tensor:
        """Normalize the pre-activation of the active branch.

        BatchNorm and `none` receive `raw` at hidden width. GraNoLa receives
        the aggregated messages at graph width, because its affine parameters
        are that wide and the out projection happens after them.
        """

        if self.normalization == "none":
            return values
        if self.normalization == "granola":
            if self.granola_adaptivity == "token":
                return self._node_layer_norm(values)
            stats = (
                prepared.norm.stats
                if isinstance(prepared.norm, _GranolaNormState)
                else None
            )
            if stats is None:
                raise ValueError("prepared graph is missing GraNoLa statistics")
            return (values - stats.mean.unsqueeze(1)) * stats.invstd.unsqueeze(1)
        if not isinstance(prepared.norm, ContextNormStats):
            raise ValueError("prepared graph is missing BatchNorm statistics")
        return (values - prepared.norm.mean.unsqueeze(1)) * prepared.norm.invstd.unsqueeze(1)

    def granola_affine(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.granola_gamma_head is None or self.granola_beta_head is None:
            raise ValueError("GraNoLa affine heads are unavailable")
        group_ids = self.normalization_group_ids(graph_ids)
        return (
            self.granola_gamma_head(hidden, group_ids),
            self.granola_beta_head(hidden, group_ids),
        )

    def activated(
        self,
        normalized: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        granola_source: Tensor | None = None,
    ) -> Tensor:
        if self.normalization == "batchnorm":
            if self.gamma is None or self.beta is None:
                raise ValueError("BatchNorm affine parameters are unavailable")
            group_ids = self.normalization_group_ids(graph_ids)
            gamma = _select_graph_rows(self.gamma, group_ids).to(normalized.dtype).unsqueeze(1)
            beta = _select_graph_rows(self.beta, group_ids).to(normalized.dtype).unsqueeze(1)
            transformed = gamma * normalized + beta
        elif self.normalization == "granola":
            if granola_source is None:
                raise ValueError("GraNoLa activation requires the GNN readout")
            gamma, beta = self.granola_affine(granola_source, graph_ids)
            transformed = gamma.to(normalized.dtype) * normalized + beta.to(normalized.dtype)
            return self.projected_activation(transformed, graph_ids)
        else:
            transformed = normalized
        return F.leaky_relu(
            transformed,
            negative_slope=self.leaky_relu_slope,
        )

    def projected_activation(
        self, transformed: Tensor, graph_ids: Sequence[int] | Tensor
    ) -> Tensor:
        """Finish a GraNoLa pre-activation: out projection, then leaky ReLU.

        The GraNoLa affine is graph width, so the out projection that the other
        branches fold into `kernel` happens here, which keeps the leaky ReLU at
        hidden width exactly as in the BatchNorm branch.
        """

        return F.leaky_relu(
            self.out_proj(transformed, graph_ids),
            negative_slope=self.leaky_relu_slope,
        )

    def delta(
        self,
        y1: Tensor,
        prepared: PreparedImplicitGraph,
        graph_ids: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        ids = prepared.graph_ids if graph_ids is None else graph_ids
        source = None
        if self.normalization == "granola":
            if not isinstance(prepared.norm, _GranolaNormState):
                raise ValueError("prepared graph is missing GraNoLa state")
            source = prepared.norm.readout()
            values = self.messages(y1, prepared.gram)
        else:
            values = self._raw(y1, prepared.kernel)
        activated = self.activated(
            self.normalized(values, prepared), ids, granola_source=source
        )
        alpha = _select_graph_rows(self.alpha, ids).to(activated.dtype).view(-1, 1, 1)
        return alpha * activated

    def forward(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        rnf_seed: int | None = None,
    ) -> Tensor:
        """Convenience full-output path for small tests; scoring uses streamed APIs."""

        prepared = self.prepare(
            hidden,
            graph_ids,
            token_microbatch_size=max(1, hidden.size(1)),
            rnf_seed=rnf_seed,
        )
        return self.delta(prepared.y1, prepared)


class _HeadwiseGateAdapter(nn.Module):
    """Apply a head-specific hidden-state delta to matching gate slices."""

    def forward(
        self, gate: nn.Module, head: int, hidden: Tensor, delta: Tensor | None
    ) -> Tensor:
        """Score one head.

        Production scoring calls `forward_batch`; this stays as the readable
        single-head statement of the same math, and the tests use it as the
        oracle `forward_batch` is checked against. Both must therefore accept
        the same inputs, including a gate-only scorer's absent delta.
        """

        token_count = hidden.size(0)
        gate_dim = gate.output_dim
        groups = gate.ngroup
        mixed = hidden if delta is None else hidden + delta

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
        delta: Tensor | None,
    ) -> Tensor:
        """Apply matching gate heads to a complete graph microbatch.

        A gate-only scorer passes no delta; the gate then reads the hidden
        states unchanged instead of adding a zero tensor of their size.
        """

        layer_ids = tuple(layer_ids)
        head_ids = tuple(head_ids)
        graph_count = len(layer_ids)
        if (
            not graph_count
            or hidden.ndim != 3
            or (delta is not None and delta.shape != hidden.shape)
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
        mixed = hidden if delta is None else hidden + delta
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
        graph_dim: int | None = 32,
        graph_microbatch_size: str | int = "auto",
        normalization: str = "batchnorm",
        normalization_sharing: str = "graph",
        granola_gnn_depth: int = 1,
        granola_mlp_depth: int = 1,
        granola_rnf_dim: int | None = None,
        granola_adaptivity: str = "graph",
        normalization_seed: int = 0,
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
        self.mixer = (
            None
            if graph_dim is None
            else ImplicitGraphMixer(
                self.num_graphs,
                self.hidden_dim,
                graph_dim,
                num_heads=self.num_heads,
                normalization=normalization,
                normalization_sharing=normalization_sharing,
                granola_gnn_depth=granola_gnn_depth,
                granola_mlp_depth=granola_mlp_depth,
                granola_rnf_dim=granola_rnf_dim,
                granola_adaptivity=granola_adaptivity,
                normalization_seed=normalization_seed,
                gram_normalization=gram_normalization,
                leaky_relu_slope=leaky_relu_slope,
                alpha_init=alpha_init,
                device=device,
                dtype=master_dtype,
            )
        )
        self._gate_adapter = _HeadwiseGateAdapter()

    @property
    def device(self) -> torch.device:
        if self.mixer is not None:
            return self.mixer.device
        # Derived rather than cached, so it survives a later .to(device).
        return self.gates[0].q_proj.weight.device

    @property
    def graph_dim(self) -> int | None:
        return None if self.mixer is None else self.mixer.graph_dim

    @property
    def uses_granola(self) -> bool:
        """Whether scoring needs a GraNoLa RNF draw. False without a mixer."""

        return self.mixer is not None and self.mixer.normalization == "granola"

    def resolve_rnf_seed(self, rnf_seed: int | None = None) -> int | None:
        """Settle one RNF seed for a whole context, or None when GraNoLa is off.

        Every pass over one context must share a seed: the graph and token
        microbatch splits must not change the scores, and an answer-training
        replay has to reproduce the forward it is backpropagating.
        """

        if not self.uses_granola:
            return None
        if rnf_seed is not None:
            return rnf_seed
        return (
            self.mixer.next_rnf_seed()
            if self.training
            else self.mixer.normalization_seed
        )

    @property
    def hidden_dtype(self) -> torch.dtype:
        """Dtype the context hidden states are materialized in for scoring.

        With a mixer the delta is accumulated in the master dtype, so adding it
        promotes the gate input to that dtype regardless of this value. A
        gate-only scorer has no delta to promote it, so it materializes the
        hidden states in the master dtype directly; otherwise dropping the
        mixer would silently drop the gate to a lower precision, confounding
        the ablation with a dtype change.
        """

        if self.mixer is not None:
            return self.compute_dtype
        # Derived rather than cached, for the same reason as `device` above: a
        # later .to(dtype) moves the gates, and a stored master dtype would go
        # stale and cast the hidden states to something they no longer match.
        return self.gates[0].q_proj.weight.dtype

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
        rnf_seed: int | None = None,
        offsets: Sequence[int] | None = None,
    ) -> PreparedImplicitGraph | None:
        if self.mixer is None:
            return None
        hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
        return self.mixer.prepare(
            hidden,
            graph_ids,
            token_microbatch_size=token_microbatch_size,
            rnf_seed=rnf_seed,
            offsets=offsets,
        )

    def score_prepared(
        self,
        hidden: Tensor,
        prepared: PreparedImplicitGraph | None,
        *,
        layer_ids: Sequence[int] | Tensor,
        head_ids: Sequence[int] | Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if (prepared is None) != (self.mixer is None):
            raise ValueError("prepared mixer state must accompany a mixer")
        if hidden.ndim != 3 or (
            prepared is not None and hidden.size(0) != len(prepared.graph_ids)
        ):
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        layer_ids = _graph_id_tuple(
            layer_ids, num_graphs=self.num_layers, expected_size=hidden.size(0)
        )
        head_ids = _graph_id_tuple(
            head_ids, num_graphs=self.num_heads, expected_size=hidden.size(0)
        )
        delta = None if prepared is None else self.mixer.delta(prepared.y1, prepared)
        scores = self._gate_adapter.forward_batch(
            self.gates,
            layer_ids,
            head_ids,
            hidden.to(device=self.device, dtype=self.hidden_dtype),
            delta,
        )
        return scores, delta

    def score_subgraph_batch(
        self,
        hidden_by_layer: Sequence[Tensor],
        batch: GraphBatch,
        starts: Sequence[int],
        length: int,
        *,
        rnf_seed: int | None = None,
    ) -> Tensor:
        """Score independent subgraphs and return [graphs, subgraphs * tokens]."""

        starts = tuple(starts)
        count = len(starts)
        # Materialize only these slices. In gradient mode, stacking also turns
        # inference-created hidden states into normal tensors for replay.
        hidden = torch.stack(
            tuple(
                hidden_by_layer[layer_id][start : start + length]
                for start in starts
                for layer_id in batch.layer_ids
            )
        ).to(device=self.device, dtype=self.hidden_dtype)
        prepared = self.prepare(
            hidden,
            batch.graph_ids * count,
            token_microbatch_size=length,
            rnf_seed=self.resolve_rnf_seed(rnf_seed),
            offsets=tuple(start for start in starts for _ in batch.graph_ids),
        )
        scores, _ = self.score_prepared(
            hidden,
            prepared,
            layer_ids=batch.layer_ids * count,
            head_ids=batch.head_ids * count,
        )
        return (
            scores.view(count, len(batch.graph_ids), length)
            .transpose(0, 1)
            .reshape(len(batch.graph_ids), -1)
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        microbatch_size: str | int | None = None,
        token_microbatch_size: int = 1000,
        rnf_seed: int | None = None,
    ) -> Tensor:
        if hidden.ndim == 4 and hidden.size(1) == 1:
            hidden = hidden[:, 0]
        if hidden.ndim != 3 or hidden.size(0) != self.num_layers:
            raise ValueError("hidden must have shape [layers,T,D] or [layers,1,T,D]")
        if hidden.size(-1) != self.hidden_dim:
            raise ValueError(f"expected hidden dimension {self.hidden_dim}")
        token_count = hidden.size(1)
        rnf_seed = self.resolve_rnf_seed(rnf_seed)
        score_batches = []
        for batch in self.graph_batches(microbatch_size=microbatch_size):
            def chunks():
                for start in range(0, token_count, token_microbatch_size):
                    stop = min(start + token_microbatch_size, token_count)
                    yield start, torch.stack(
                        tuple(hidden[layer_id, start:stop] for layer_id in batch.layer_ids)
                    ).to(device=self.device, dtype=self.compute_dtype)

            prepared = (
                None
                if self.mixer is None
                else self.mixer.prepare_from_chunks(
                    chunks(),
                    graph_ids=batch.graph_ids,
                    token_count=token_count,
                    token_microbatch_size=token_microbatch_size,
                    rnf_seed=rnf_seed,
                )
            )
            chunks_scores = []
            for start in range(0, token_count, token_microbatch_size):
                stop = min(start + token_microbatch_size, token_count)
                graph_hidden = torch.stack(
                    tuple(hidden[layer_id, start:stop] for layer_id in batch.layer_ids)
                ).to(device=self.device, dtype=self.hidden_dtype)
                slice_prepared = (
                    None
                    if prepared is None
                    else prepared.select_tokens(
                        torch.arange(start, stop, device=prepared.y1.device)
                    )
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
