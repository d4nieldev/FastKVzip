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
GPS_ACTIVATION_ORDER = "gps-layernorm-gelu-leaky-relu"

NORMALIZATIONS = ("none", "batchnorm", "granola")
NORMALIZATION_SHARING = ("graph", "layer", "global")
GRANOLA_ADAPTIVITY = ("graph", "token")

MIXER_ARCHITECTURES = ("implicit", "gps")
DEFAULT_MIXER_ARCHITECTURE = "implicit"

# How the mixer's output reaches the gate. `hidden` adds a hidden-width
# residual to the gate's input, the original design. `gate-space` keeps the
# features at graph width and adds zero-initialized maps of them to the gate's
# normalized queries and keys, and optionally to its per-group logit bias.
MIXER_COUPLINGS = ("hidden", "gate-space")
DEFAULT_MIXER_COUPLING = "hidden"
INJECTION_TARGETS = ("qk", "logit", "qk-logit")
DEFAULT_INJECTION_TARGET = "qk-logit"
# Standard deviation the injection maps start at. Zero starts the run exactly
# at the gate's own scores, but the mixer behind the maps then has no gradient
# until they grow, because its gradient arrives through them. A small nonzero
# value trades that exact start for a mixer that trains from the first step.
DEFAULT_INJECTION_INIT = 0.0

GPS_DEFAULT_ATTENTION_HEADS = 4
GPS_DEFAULT_RANDOM_FEATURES = 32
# Fixed rather than exposed: one more knob per architecture buys little next to
# graph width, which already controls the block's size.
GPS_FFN_MULTIPLIER = 2

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
# reads it without a mixer, but it still has to be a usable width. A quarter of
# the default graph width, which is what a mixer picks when told nothing.
DEFAULT_GRANOLA_RNF_DIM = 8


def parse_mixer_architecture(value: object) -> str:
    if value not in MIXER_ARCHITECTURES:
        raise ValueError(
            f"mixer architecture must be one of {', '.join(MIXER_ARCHITECTURES)}"
        )
    return str(value)


def parse_mixer_coupling(value: object) -> str:
    if value not in MIXER_COUPLINGS:
        raise ValueError(f"mixer coupling must be one of {', '.join(MIXER_COUPLINGS)}")
    return str(value)


def parse_injection_target(value: object) -> str:
    if value not in INJECTION_TARGETS:
        raise ValueError(
            f"injection target must be one of {', '.join(INJECTION_TARGETS)}"
        )
    return str(value)


def mixer_activation_order(architecture: str) -> str:
    """Activation order recorded for, and validated against, one architecture.

    The implicit mixer's order is the same whichever normalization it uses, so
    it keeps one name. A GPS stack normalizes inside its own blocks and applies
    its activation in a different place, so it gets its own.
    """

    return (
        ACTIVATION_ORDER
        if parse_mixer_architecture(architecture) == "implicit"
        else GPS_ACTIVATION_ORDER
    )


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
            # Matches the live default. A checkpoint reaching this branch
            # predates the normalization setting, so it is BatchNorm and no
            # GraNoLa width applies to it; the value exists only so the
            # consistency check against a default-built scorer passes.
            "granola_rnf_dim": (
                DEFAULT_GRANOLA_RNF_DIM if graph_dim is None else max(1, graph_dim // 4)
            ),
            "granola_adaptivity": "graph",
            "normalization_seed": 0,
        }
    )
    if canonical.get("activation_order") == LEGACY_ACTIVATION_ORDER:
        canonical["activation_order"] = ACTIVATION_ORDER
    return canonical


def canonical_checkpoint_config(config: Mapping[str, object]) -> dict[str, object]:
    """Fill in every setting an older checkpoint lacks, normalization included.

    The architecture is filled here rather than inside the normalization helper
    because that one returns early for a checkpoint that already names a
    normalization -- and a checkpoint can name one while still predating the
    architecture becoming a choice.

    A checkpoint with a mixer but no architecture holds an implicit one. A
    gate-only checkpoint has no mixer to name. The GPS settings need no default:
    only a GPS run records them, so neither side of a comparison has them.
    """

    canonical = dict(config)
    if canonical.get("graph_dim") is not None:
        canonical.setdefault("mixer_architecture", DEFAULT_MIXER_ARCHITECTURE)
        # Every checkpoint written before the coupling became a choice adds a
        # hidden-width residual. The gate-space settings need no default: only
        # a gate-space run records them.
        canonical.setdefault("mixer_coupling", DEFAULT_MIXER_COUPLING)
    return canonical_normalization_config(canonical)


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
class GateInjection:
    """Graph-width mixer features mapped into the gate's own spaces.

    Each field is `[graphs, tokens, ...]`, or None when the injection target
    leaves that part of the gate alone. `query` and `key` are gate width and
    are added after the gate's RMSNorms; `logit` holds one bias per query
    group and is added next to the gate's own bias.
    """

    query: Tensor | None
    key: Tensor | None
    logit: Tensor | None

    def _map(self, function) -> "GateInjection":
        return GateInjection(
            None if self.query is None else function(self.query),
            None if self.key is None else function(self.key),
            None if self.logit is None else function(self.logit),
        )

    @property
    def tokens(self) -> int:
        for value in (self.query, self.key, self.logit):
            if value is not None:
                return value.size(1)
        raise ValueError("an injection must carry at least one tensor")

    def select_tokens(self, index: Tensor) -> "GateInjection":
        return self._map(lambda value: value.index_select(1, index.to(value.device)))

    def detached_to(self, device: str | torch.device) -> "GateInjection":
        return self._map(lambda value: value.detach().to(device))


class GateSpaceInjection(nn.Module):
    """Zero-initialized per-graph maps from mixer features into the gate.

    With every map at zero the gate scores exactly as it does alone, so a run
    starts from the released gate and the mixer only grows where the loss asks
    for it. The maps are the only parameters of the gate-space coupling; the
    module lives on the mixer so checkpoints and optimizers reach it as mixer
    state.
    """

    def __init__(
        self,
        num_graphs: int,
        graph_dim: int,
        *,
        gate_dim: int,
        query_groups: int,
        target: str = DEFAULT_INJECTION_TARGET,
        init_std: float = DEFAULT_INJECTION_INIT,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if gate_dim < 1 or query_groups < 1:
            raise ValueError("gate_dim and query_groups must be positive")
        if (
            isinstance(init_std, bool)
            or not isinstance(init_std, (int, float))
            or not math.isfinite(init_std)
            or init_std < 0
        ):
            raise ValueError("injection init must be finite and non-negative")
        self.target = parse_injection_target(target)
        self.init_std = float(init_std)
        self.query_proj = None
        self.key_proj = None
        self.logit_proj = None
        if "qk" in self.target:
            self.query_proj = PerGraphLinear(
                num_graphs, graph_dim, gate_dim, device=device, dtype=dtype
            )
            self.key_proj = PerGraphLinear(
                num_graphs, graph_dim, gate_dim, device=device, dtype=dtype
            )
        if "logit" in self.target:
            self.logit_proj = PerGraphLinear(
                num_graphs, graph_dim, query_groups, device=device, dtype=dtype
            )
        with torch.no_grad():
            for parameter in self.parameters():
                if self.init_std:
                    parameter.normal_(std=self.init_std)
                else:
                    parameter.zero_()

    def forward(
        self, features: Tensor, graph_ids: Sequence[int] | Tensor
    ) -> GateInjection:
        if features.ndim != 3:
            raise ValueError("features must have shape [graphs,tokens,graph_dim]")
        return GateInjection(
            None if self.query_proj is None else self.query_proj(features, graph_ids),
            None if self.key_proj is None else self.key_proj(features, graph_ids),
            None if self.logit_proj is None else self.logit_proj(features, graph_ids),
        )


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
        coupling: str = DEFAULT_MIXER_COUPLING,
        injection_target: str = DEFAULT_INJECTION_TARGET,
        injection_init: float = DEFAULT_INJECTION_INIT,
        self_loop_init: float | None = None,
        gate_dim: int | None = None,
        query_groups: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or hidden_dim < 1 or graph_dim < 1 or num_heads < 1:
            raise ValueError(
                "num_graphs, hidden_dim, graph_dim, and num_heads must be positive"
            )
        coupling = parse_mixer_coupling(coupling)
        if coupling == "gate-space" and (gate_dim is None or query_groups is None):
            raise ValueError(
                "gate-space coupling needs the gate dimension and query groups"
            )
        if self_loop_init is not None and (
            isinstance(self_loop_init, bool)
            or not isinstance(self_loop_init, (int, float))
            or not math.isfinite(self_loop_init)
        ):
            raise ValueError("self_loop_init must be finite")
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
            # A quarter of the graph width. The random features are projected
            # by their own square MLP before the GNN sees them, so they do not
            # need to match the message features to be useful. Floored at one
            # so a narrow graph still draws a feature.
            granola_rnf_dim = max(1, graph_dim // 4)
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
        self.coupling = coupling
        self.self_loop_init = None if self_loop_init is None else float(self_loop_init)
        self.in_proj = PerGraphLinear(
            num_graphs, hidden_dim, 2 * graph_dim, device=device, dtype=dtype
        )
        # The hidden coupling projects back to hidden width and scales the
        # residual by alpha. The gate-space coupling has neither: its features
        # stay at graph width and reach the gate through the injection maps.
        if coupling == "hidden":
            self.out_proj = PerGraphLinear(
                num_graphs, graph_dim, hidden_dim, device=device, dtype=dtype
            )
            self.alpha = nn.Parameter(
                torch.full((num_graphs,), alpha_init, device=device, dtype=dtype)
            )
            self.injection = None
        else:
            self.out_proj = None
            self.register_parameter("alpha", None)
            self.injection = GateSpaceInjection(
                num_graphs,
                graph_dim,
                gate_dim=gate_dim,
                query_groups=query_groups,
                target=injection_target,
                init_std=injection_init,
                device=device,
                dtype=dtype,
            )
        affine_width = hidden_dim if coupling == "hidden" else graph_dim
        if normalization == "batchnorm":
            self.gamma = nn.Parameter(
                torch.ones(
                    self.num_normalization_groups,
                    affine_width,
                    device=device,
                    dtype=dtype,
                )
            )
            self.beta = nn.Parameter(
                torch.zeros(
                    self.num_normalization_groups,
                    affine_width,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        # One learnable self-loop weight per graph, only when asked for, so a
        # run without the flag keeps the original adjacency and its checkpoints.
        if self_loop_init is None:
            self.register_parameter("self_loop", None)
        else:
            self.self_loop = nn.Parameter(
                torch.full(
                    (num_graphs,), float(self_loop_init), device=device, dtype=dtype
                )
            )
        self.granola_blocks = nn.ModuleList()
        self.granola_gamma_head = None
        self.granola_beta_head = None
        self.granola_rnf_mlp = None
        if normalization == "granola":
            # The random features are learned into before the GNN sees them.
            # Square, so the concatenated width the first block expects is
            # unchanged.
            self.granola_rnf_mlp = _PerGroupMLP(
                self.num_normalization_groups,
                granola_rnf_dim,
                granola_rnf_dim,
                granola_rnf_dim,
                granola_mlp_depth,
                bias=False,
                device=device,
                dtype=dtype,
            )
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

    def _self_loop_rows(
        self, graph_ids: Sequence[int] | Tensor, dtype: torch.dtype
    ) -> Tensor | None:
        if self.self_loop is None:
            return None
        return _select_graph_rows(self.self_loop, graph_ids).to(dtype).view(-1, 1, 1)

    def message(
        self,
        y1: Tensor,
        y2: Tensor | None,
        gram: Tensor,
        graph_ids: Sequence[int] | Tensor,
    ) -> Tensor:
        """Aggregated messages with the optional self-loop term, at graph width.

        `Y1 (Y1^T Y2 / T) + lambda Y2`: the implicit adjacency `Y1 Y1^T / T`
        plus `lambda I`, never materialized.
        """

        values = self.messages(y1, gram)
        weight = self._self_loop_rows(graph_ids, values.dtype)
        if weight is None:
            return values
        if y2 is None:
            raise ValueError("self loops require the retained message features")
        return values + weight * y2.to(values.dtype)

    def _raw_with_self_loop(
        self,
        y1: Tensor,
        y2: Tensor | None,
        kernel: Tensor,
        graph_ids: Sequence[int] | Tensor,
    ) -> Tensor:
        """Hidden-width pre-activation `(Y1 S + lambda Y2) W`.

        Folded as `Y1 K + lambda (Y2 W^T)` with the existing kernel `K = S W^T`,
        so the self-loop term costs one extra product per chunk.
        """

        raw = self._raw(y1, kernel)
        weight = self._self_loop_rows(graph_ids, raw.dtype)
        if weight is None:
            return raw
        if y2 is None:
            raise ValueError("self loops require the retained message features")
        out_weight = _select_graph_rows(self.out_proj.weight, graph_ids).to(raw.dtype)
        return raw + weight * torch.bmm(y2.to(raw.dtype), out_weight.transpose(1, 2))

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
        if self.granola_rnf_mlp is None:
            raise ValueError("mixer is not configured for GraNoLa")
        values = torch.cat(
            (y2.to(dtype), self.granola_rnf_mlp(rnf.to(dtype), group_ids)), dim=-1
        )
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
        return self.context_normalized(messages)

    def context_normalized(self, values: Tensor) -> Tensor:
        """Normalize over the tokens of one context from statistics computed here.

        Scoring reuses the statistics stored when the graph was prepared; the
        trainer's live graph-width pass recomputes them so it can differentiate
        through them. Both use the same definition, so the two cannot drift.
        """

        stats = self._context_norm_stats(iter((values,)))
        return (values - stats.mean.unsqueeze(1)) * stats.invstd.unsqueeze(1)

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
            stats=self._context_norm_stats(
                iter((self.message(y1, y2, gram, graph_ids),))
            ),
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
        keep_y2 = (
            self.normalization == "granola"
            or self.self_loop is not None
            or self.coupling == "gate-space"
        )
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
        # runs at graph width, before the out projection this folds in. The
        # gate-space coupling has no out projection at all.
        kernel = (
            None
            if self.normalization == "granola" or self.coupling == "gate-space"
            else self._kernel(gram, graph_ids)
        )
        if self.normalization == "batchnorm":
            if self.coupling == "gate-space":
                stream = (
                    self.message(
                        y1[:, start:stop],
                        None if y2 is None else y2[:, start:stop],
                        gram,
                        graph_ids,
                    )
                    for start, stop in self._chunks(token_count, token_microbatch_size)
                )
            else:
                stream = (
                    self._raw_with_self_loop(
                        y1[:, start:stop],
                        None if y2 is None else y2[:, start:stop],
                        kernel,
                        graph_ids,
                    )
                    for start, stop in self._chunks(token_count, token_microbatch_size)
                )
            norm: ContextNormStats | _GranolaNormState | None = self._context_norm_stats(
                stream
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
            if self.coupling == "hidden":
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
        if self.coupling != "hidden":
            raise ValueError("the gate-space coupling produces an injection, not a delta")
        ids = prepared.graph_ids if graph_ids is None else graph_ids
        source = None
        if self.normalization == "granola":
            if not isinstance(prepared.norm, _GranolaNormState):
                raise ValueError("prepared graph is missing GraNoLa state")
            source = prepared.norm.readout()
            values = self.message(y1, prepared.y2, prepared.gram, ids)
        else:
            values = self._raw_with_self_loop(y1, prepared.y2, prepared.kernel, ids)
        activated = self.activated(
            self.normalized(values, prepared), ids, granola_source=source
        )
        alpha = _select_graph_rows(self.alpha, ids).to(activated.dtype).view(-1, 1, 1)
        return alpha * activated

    def features(
        self,
        y1: Tensor,
        y2: Tensor | None,
        prepared: PreparedImplicitGraph,
        graph_ids: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        """Graph-width activated features, what the gate-space coupling injects."""

        if self.coupling != "gate-space":
            raise ValueError("the hidden coupling produces a delta, not features")
        ids = prepared.graph_ids if graph_ids is None else graph_ids
        source = None
        if self.normalization == "granola":
            if not isinstance(prepared.norm, _GranolaNormState):
                raise ValueError("prepared graph is missing GraNoLa state")
            source = prepared.norm.readout()
        values = self.message(y1, y2, prepared.gram, ids)
        return self.activated(
            self.normalized(values, prepared), ids, granola_source=source
        )

    def correction_from_prepared(
        self, prepared: PreparedImplicitGraph
    ) -> Tensor | GateInjection:
        """The gate correction this coupling produces: a delta or an injection."""

        if not isinstance(prepared, PreparedImplicitGraph):
            raise ValueError("the implicit mixer requires prepared implicit state")
        if self.coupling == "hidden":
            return self.delta(prepared.y1, prepared)
        return self.injection(
            self.features(prepared.y1, prepared.y2, prepared), prepared.graph_ids
        )

    def coupling_config(self) -> dict[str, object]:
        """The coupling settings this mixer applies, recorded only when applied."""

        config: dict[str, object] = {"mixer_coupling": self.coupling}
        if self.injection is not None:
            config["injection_target"] = self.injection.target
            config["injection_init"] = self.injection.init_std
        if self.self_loop is not None:
            config["self_loop_init"] = self.self_loop_init
        return config

    def parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        """Split parameters into weight-decayed and undecayed groups.

        Which parameters exist depends on the normalization: BatchNorm keeps a
        scale and shift, GraNoLa predicts them with a small network, and none
        has neither.
        """

        decay = [self.in_proj.weight]
        no_decay = []
        if self.out_proj is not None:
            decay.append(self.out_proj.weight)
        if self.injection is not None:
            decay.extend(self.injection.parameters())
        if self.alpha is not None:
            no_decay.append(self.alpha)
        if self.self_loop is not None:
            no_decay.append(self.self_loop)
        if self.normalization == "batchnorm":
            no_decay.extend((self.gamma, self.beta))
        elif self.normalization == "granola":
            for name, parameter in self.named_parameters():
                if not name.startswith(
                    (
                        "granola_blocks.",
                        "granola_gamma_head.",
                        "granola_beta_head.",
                        "granola_rnf_mlp.",
                    )
                ):
                    continue
                # Only the linear weights decay. Their biases, and every
                # normalization scale and shift, stay undecayed.
                target = (
                    decay
                    if ".linears." in name and name.endswith(".weight")
                    else no_decay
                )
                target.append(parameter)
        return decay, no_decay

    def normalization_config(self) -> dict[str, object]:
        """The normalization settings this mixer applies."""

        return {
            "normalization": self.normalization,
            "normalization_sharing": self.normalization_sharing,
            "granola_gnn_depth": self.granola_gnn_depth,
            "granola_mlp_depth": self.granola_mlp_depth,
            "granola_rnf_dim": self.granola_rnf_dim,
            "granola_adaptivity": self.granola_adaptivity,
            "normalization_seed": self.normalization_seed,
        }

    def on_optimizer_step(self) -> None:
        """Nothing here is resampled on an optimizer step."""

    def delta_from_prepared(self, prepared: PreparedImplicitGraph) -> Tensor:
        if not isinstance(prepared, PreparedImplicitGraph):
            raise ValueError("the implicit mixer requires prepared implicit state")
        return self.delta(prepared.y1, prepared)

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
        return self.correction_from_prepared(prepared)


class PerGraphLayerNorm(nn.Module):
    """Layer normalization with independent scale and shift per graph."""

    def __init__(
        self,
        num_graphs: int,
        features: int,
        *,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.features = features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_graphs, features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_graphs, features, device=device, dtype=dtype))

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        normalized = F.layer_norm(x, (self.features,), eps=self.eps)
        weight = _select_graph_rows(self.weight, graph_ids).to(x.dtype).unsqueeze(1)
        bias = _select_graph_rows(self.bias, graph_ids).to(x.dtype).unsqueeze(1)
        return normalized * weight + bias


_POSITION_CACHE: dict[tuple, Tensor] = {}


def sinusoidal_positions(
    token_count: int, features: int, *, device, dtype: torch.dtype
) -> Tensor:
    """Sequence position encoding over a subgraph's token order.

    GPS expects positional information on its input. The graph is complete, so
    spectral encodings are degenerate here, but the tokens carry their sequence
    order; this encodes the token's index inside its own subgraph.
    """

    if token_count < 1 or features < 1:
        raise ValueError("token_count and features must be positive")
    key = (token_count, features, str(device), dtype)
    cached = _POSITION_CACHE.get(key)
    if cached is not None:
        return cached
    position = torch.arange(token_count, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(features, device=device, dtype=torch.float32)
    # index // 2 pairs each sine with its cosine, and leaves an odd width valid.
    angles = position * torch.exp(
        -math.log(10000.0) * (2 * torch.div(index, 2, rounding_mode="floor")) / features
    )
    # Fill alternating columns rather than computing both functions everywhere
    # and discarding half of each.
    encoding = torch.empty_like(angles)
    encoding[:, 0::2] = angles[:, 0::2].sin()
    encoding[:, 1::2] = angles[:, 1::2].cos()
    encoding = encoding.to(dtype)
    # Depends only on the key, so one entry per subgraph length is all it holds.
    if len(_POSITION_CACHE) >= 8:
        _POSITION_CACHE.clear()
    _POSITION_CACHE[key] = encoding
    return encoding


def orthogonal_random_features(
    rows: int, columns: int, *, device, dtype: torch.dtype, draws: int = 1
) -> Tensor:
    """Draw FAVOR+ orthogonal random features, `draws` independent sets at once.

    Returns [rows, columns] for a single draw, else [draws, rows, columns]. The
    factorizations are batched: a stack has one set per graph and head, and a
    per-draw Python loop costs thousands of tiny factorizations per block.
    """

    if rows < 1 or columns < 1 or draws < 1:
        raise ValueError("rows, columns, and draws must be positive")
    blocks = -(-rows // columns)
    gaussian = torch.randn(draws, blocks, columns, columns, device=device, dtype=dtype)
    orthogonal, upper = torch.linalg.qr(gaussian)
    # QR alone is not Haar-uniform: its sign convention leaves the directions
    # non-uniform on the sphere, which biases the kernel estimate. Folding in
    # the signs of R's diagonal restores uniformity.
    signs = torch.sign(torch.diagonal(upper, dim1=-2, dim2=-1)).unsqueeze(-2)
    directions = (orthogonal * signs).transpose(-2, -1).reshape(
        draws, blocks * columns, columns
    )[:, :rows]
    # Orthogonal directions with chi-distributed lengths, as FAVOR+ specifies.
    lengths = torch.randn(draws, rows, columns, device=device, dtype=dtype).norm(dim=-1)
    features = lengths.unsqueeze(-1) * directions
    return features if draws > 1 else features[0]


class _PerGraphPerformerAttention(nn.Module):
    """Softmax attention approximated by positive orthogonal random features.

    This is the GPS global branch. It differs from the implicit branch beside
    it by untying query from key, normalizing by the attention denominator, and
    splitting the width across heads.
    """

    def __init__(
        self,
        num_graphs: int,
        graph_dim: int,
        heads: int,
        random_features: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if heads < 1 or graph_dim % heads:
            raise ValueError("graph_dim must be a positive multiple of the head count")
        if random_features < 1:
            raise ValueError("random_features must be positive")
        self.heads = heads
        self.head_dim = graph_dim // heads
        self.random_features = random_features
        self.qkv_proj = PerGraphLinear(
            num_graphs, graph_dim, 3 * graph_dim, device=device, dtype=dtype
        )
        self.out_proj = PerGraphLinear(
            num_graphs, graph_dim, graph_dim, device=device, dtype=dtype
        )
        # Drawn once and registered, so the checkpoint reproduces the scores the
        # run was trained to produce. FAVOR+ redraws periodically; fixing them
        # keeps evaluation deterministic.
        self.num_graphs = num_graphs
        self.register_buffer("projection", self._draw(device, dtype))

    def _draw(self, device=None, dtype=None) -> Tensor:
        """One independent feature set per graph and head."""

        weight = self.qkv_proj.weight
        features = orthogonal_random_features(
            self.random_features,
            self.head_dim,
            device=weight.device if device is None else device,
            dtype=torch.float32,
            draws=self.num_graphs * self.heads,
        )
        return features.view(
            self.num_graphs, self.heads, self.random_features, self.head_dim
        ).to(dtype=weight.dtype if dtype is None else dtype)

    @torch.no_grad()
    def redraw(self) -> None:
        """Replace the features with a fresh independent draw."""

        self.projection.copy_(self._draw(self.projection.device, self.projection.dtype))

    def _features(self, values: Tensor, projection: Tensor, *, per_token: bool) -> Tensor:
        """Positive random features phi(x), stabilized against overflow.

        Subtracting a maximum before the exponential cancels between the
        attention numerator and denominator, so any choice that is constant
        across the summed axis is exact. Queries use a per-token maximum and
        keys a maximum over the whole span, which is what keeps a token whose
        scores sit far below the span's maximum from underflowing in bf16.
        """

        scores = torch.einsum("gthd,ghmd->gthm", values, projection)
        squared = values.square().sum(dim=-1, keepdim=True) / 2
        dims = (3,) if per_token else (1, 3)
        stabilizer = torch.amax(scores - squared, dim=dims, keepdim=True).detach()
        return torch.exp(scores - squared - stabilizer) / math.sqrt(self.random_features)

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        graphs, tokens, _ = x.shape
        packed = self.qkv_proj(x, graph_ids)
        query, key, value = (
            part.view(graphs, tokens, self.heads, self.head_dim)
            for part in packed.split(packed.size(-1) // 3, dim=-1)
        )
        # Fold the softmax temperature into the inputs, as Performer does.
        scale = self.head_dim ** -0.25
        projection = _select_graph_rows(self.projection, graph_ids).to(x.dtype)
        query_features = self._features(query * scale, projection, per_token=True)
        key_features = self._features(key * scale, projection, per_token=False)
        context = torch.einsum("gthm,gthd->ghmd", key_features, value)
        numerator = torch.einsum("gthm,ghmd->gthd", query_features, context)
        normalizer = torch.einsum(
            "gthm,ghm->gth", query_features, key_features.sum(dim=1)
        )
        # The stabilizers leave the denominator on no fixed scale, so an absolute
        # floor would clamp healthy values and distort the result. Every feature
        # is a positive exponential, so the denominator can only reach zero by
        # underflowing; guard exactly that.
        floor = torch.finfo(normalizer.dtype).tiny
        attended = numerator / normalizer.clamp_min(floor).unsqueeze(-1)
        return self.out_proj(attended.reshape(graphs, tokens, -1), graph_ids)


class _PerGraphImplicitBranch(nn.Module):
    """The existing low-rank aggregation, applied inside the graph space.

    Same mechanism as `ImplicitGraphMixer`: a similarity kernel built from tied
    projections, aggregating a second projection. Only the projection back to
    hidden width is missing, because a GPS stack does that once at the end.
    """

    def __init__(
        self,
        num_graphs: int,
        graph_dim: int,
        *,
        gram_normalization: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if gram_normalization not in {"token-count", "none"}:
            raise ValueError("gram_normalization must be token-count or none")
        self.graph_dim = graph_dim
        self.gram_normalization = gram_normalization
        self.proj = PerGraphLinear(
            num_graphs, graph_dim, 2 * graph_dim, device=device, dtype=dtype
        )

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        first, second = self.proj(x, graph_ids).split(self.graph_dim, dim=-1)
        # Accumulate the Gram sum in the reduction dtype, as the implicit mixer
        # does: it sums over every token in the subgraph. Upcast once; both uses
        # below need the same tensor.
        dtype = _reduction_dtype(first, second)
        first = first.to(dtype)
        gram = torch.bmm(first.transpose(1, 2), second.to(dtype))
        if self.gram_normalization == "token-count":
            gram = gram / x.size(1)
        return torch.bmm(first, gram).to(x.dtype)


class _GPSBlock(nn.Module):
    """One GPS layer: two branches, each normalized, then a feedforward."""

    def __init__(
        self,
        num_graphs: int,
        graph_dim: int,
        *,
        attention_heads: int,
        random_features: int,
        gram_normalization: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.local = _PerGraphImplicitBranch(
            num_graphs,
            graph_dim,
            gram_normalization=gram_normalization,
            device=device,
            dtype=dtype,
        )
        self.attention = _PerGraphPerformerAttention(
            num_graphs,
            graph_dim,
            attention_heads,
            random_features,
            device=device,
            dtype=dtype,
        )
        self.local_norm = PerGraphLayerNorm(num_graphs, graph_dim, device=device, dtype=dtype)
        self.attention_norm = PerGraphLayerNorm(
            num_graphs, graph_dim, device=device, dtype=dtype
        )
        self.ffn_norm = PerGraphLayerNorm(num_graphs, graph_dim, device=device, dtype=dtype)
        inner = GPS_FFN_MULTIPLIER * graph_dim
        self.ffn_in = PerGraphLinear(num_graphs, graph_dim, inner, device=device, dtype=dtype)
        self.ffn_out = PerGraphLinear(num_graphs, inner, graph_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        local = self.local_norm(x + self.local(x, graph_ids), graph_ids)
        attended = self.attention_norm(x + self.attention(x, graph_ids), graph_ids)
        merged = local + attended
        inner = F.gelu(self.ffn_in(merged, graph_ids))
        return self.ffn_norm(merged + self.ffn_out(inner, graph_ids), graph_ids)


@dataclass(frozen=True)
class PreparedGPSGraph:
    """A GPS stack's finished gate correction for one token span.

    Under the hidden coupling this is the hidden-state delta; under the
    gate-space coupling it is the injection into the gate.
    """

    graph_ids: tuple[int, ...]
    correction: Tensor | GateInjection

    @property
    def delta(self) -> Tensor:
        """The hidden-width delta; only the hidden coupling has one."""

        if not isinstance(self.correction, Tensor):
            raise ValueError("the gate-space coupling produces an injection, not a delta")
        return self.correction

    @property
    def tokens(self) -> int:
        if isinstance(self.correction, GateInjection):
            return self.correction.tokens
        return self.correction.size(1)

    def select_tokens(self, index: Tensor) -> "PreparedGPSGraph":
        # Selecting every token in order is what validation always asks for, and
        # index_select would copy the whole correction to answer it.
        if index.numel() == self.tokens and bool(
            torch.equal(index.cpu(), torch.arange(index.numel()))
        ):
            return self
        if isinstance(self.correction, GateInjection):
            return PreparedGPSGraph(self.graph_ids, self.correction.select_tokens(index))
        return PreparedGPSGraph(
            self.graph_ids,
            self.correction.index_select(1, index.to(self.correction.device)),
        )

    def detached_to(self, device: str | torch.device) -> "PreparedGPSGraph":
        if isinstance(self.correction, GateInjection):
            return PreparedGPSGraph(self.graph_ids, self.correction.detached_to(device))
        return PreparedGPSGraph(self.graph_ids, self.correction.detach().to(device))


class GPSGraphMixer(nn.Module):
    """A GPS stack per graph, over one fixed-size subgraph at a time.

    It produces the same kind of hidden-state correction the implicit mixer
    produces, so nothing downstream of the mixer changes. Unlike the implicit
    mixer it keeps every token's activations, so it scores subgraphs rather
    than a whole context.
    """

    def __init__(
        self,
        num_graphs: int,
        hidden_dim: int,
        graph_dim: int,
        *,
        depth: int = 1,
        attention_heads: int = GPS_DEFAULT_ATTENTION_HEADS,
        random_features: int = GPS_DEFAULT_RANDOM_FEATURES,
        redraw_interval: int = 0,
        gram_normalization: str = "token-count",
        leaky_relu_slope: float = 0.01,
        alpha_init: float = 0.1,
        coupling: str = DEFAULT_MIXER_COUPLING,
        injection_target: str = DEFAULT_INJECTION_TARGET,
        injection_init: float = DEFAULT_INJECTION_INIT,
        gate_dim: int | None = None,
        query_groups: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or hidden_dim < 1 or graph_dim < 1:
            raise ValueError("num_graphs, hidden_dim, and graph_dim must be positive")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("gps depth must be a positive integer")
        coupling = parse_mixer_coupling(coupling)
        if coupling == "gate-space" and (gate_dim is None or query_groups is None):
            raise ValueError(
                "gate-space coupling needs the gate dimension and query groups"
            )
        if not math.isfinite(leaky_relu_slope) or leaky_relu_slope < 0:
            raise ValueError("leaky_relu_slope must be finite and non-negative")
        if not math.isfinite(alpha_init):
            raise ValueError("alpha_init must be finite")
        if isinstance(redraw_interval, bool) or not isinstance(redraw_interval, int) or redraw_interval < 0:
            raise ValueError("redraw_interval must be a non-negative integer")
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.depth = depth
        self.attention_heads = attention_heads
        self.random_features = random_features
        self.redraw_interval = redraw_interval
        # A GPS stack normalizes inside its own blocks, so no mixer-level
        # normalization applies. Saying so keeps the shared guards -- which ask
        # the mixer which normalization it uses -- answering correctly.
        self.normalization = "none"
        self.gram_normalization = gram_normalization
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.coupling = coupling
        self.in_proj = PerGraphLinear(
            num_graphs, hidden_dim, graph_dim, device=device, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            _GPSBlock(
                num_graphs,
                graph_dim,
                attention_heads=attention_heads,
                random_features=random_features,
                gram_normalization=gram_normalization,
                device=device,
                dtype=dtype,
            )
            for _ in range(depth)
        )
        # As for the implicit mixer: the hidden coupling projects back to hidden
        # width and scales by alpha, the gate-space coupling injects instead.
        if coupling == "hidden":
            self.out_proj = PerGraphLinear(
                num_graphs, graph_dim, hidden_dim, device=device, dtype=dtype
            )
            self.alpha = nn.Parameter(
                torch.full((num_graphs,), alpha_init, device=device, dtype=dtype)
            )
            self.injection = None
        else:
            self.out_proj = None
            self.register_parameter("alpha", None)
            self.injection = GateSpaceInjection(
                num_graphs,
                graph_dim,
                gate_dim=gate_dim,
                query_groups=query_groups,
                target=injection_target,
                init_std=injection_init,
                device=device,
                dtype=dtype,
            )
        # Registered, so a resumed run continues on the same redraw schedule the
        # interrupted one was following.
        self.register_buffer(
            "redraw_step", torch.zeros((), device=device, dtype=torch.long)
        )

    @property
    def device(self) -> torch.device:
        return self.in_proj.weight.device

    def on_optimizer_step(self) -> None:
        """Redraw the random features every `redraw_interval` optimizer steps.

        FAVOR+ resamples periodically: with one frozen draw its approximation
        error is a fixed distortion the model can fit, rather than noise that
        averages out. Counting optimizer steps rather than forward calls keeps
        the schedule independent of the memory knobs, which change how many
        forwards one step makes.
        """

        if not self.training or not self.redraw_interval:
            return
        self.redraw_step += 1
        if int(self.redraw_step) % self.redraw_interval == 0:
            for block in self.blocks:
                block.attention.redraw()

    def normalization_config(self) -> dict[str, object]:
        """Only the mode: a GPS stack applies no mixer-level normalization."""

        return {"normalization": self.normalization}

    def parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        """Split parameters into weight-decayed and undecayed groups."""

        no_decay = [] if self.alpha is None else [self.alpha]
        for module in self.modules():
            if isinstance(module, PerGraphLayerNorm):
                no_decay.extend((module.weight, module.bias))
        undecayed = {id(parameter) for parameter in no_decay}
        decay = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in undecayed
        ]
        return decay, no_decay

    def _stack(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        """Run the GPS stack; the graph-width output both couplings start from."""

        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        x = self.in_proj(hidden, graph_ids)
        x = x + sinusoidal_positions(
            x.size(1), self.graph_dim, device=x.device, dtype=x.dtype
        )
        for block in self.blocks:
            x = block(x, graph_ids)
        return x

    def delta(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        if self.coupling != "hidden":
            raise ValueError("the gate-space coupling produces an injection, not a delta")
        projected = self.out_proj(self._stack(hidden, graph_ids), graph_ids)
        # Return the delta in the reduction dtype, as the implicit mixer does.
        # Adding it is what promotes the gate's input above the compute dtype,
        # and the gate's own normalization returns that higher precision either
        # way; a delta left in bfloat16 makes the two disagree.
        dtype = _reduction_dtype(projected)
        alpha = _select_graph_rows(self.alpha, graph_ids).to(dtype).view(-1, 1, 1)
        return alpha * F.leaky_relu(
            projected.to(dtype), negative_slope=self.leaky_relu_slope
        )

    def correction(
        self, hidden: Tensor, graph_ids: Sequence[int] | Tensor
    ) -> Tensor | GateInjection:
        """The gate correction this coupling produces: a delta or an injection."""

        if self.coupling == "hidden":
            return self.delta(hidden, graph_ids)
        features = self._stack(hidden, graph_ids)
        dtype = _reduction_dtype(features)
        return self.injection(
            F.leaky_relu(features.to(dtype), negative_slope=self.leaky_relu_slope),
            graph_ids,
        )

    def prepare(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        token_microbatch_size: int | None = None,
        rnf_seed: int | None = None,
        offsets: Sequence[int] | None = None,
    ) -> PreparedGPSGraph:
        """Run the stack once. Token microbatching does not apply to GPS.

        `rnf_seed` and `offsets` belong to GraNoLa's seeded feature draw. They
        are accepted because callers forward them without asking which mixer
        they have, and ignored because this one draws nothing here.

        The implicit mixer streams a context in token microbatches because its
        state is a fixed-size Gram matrix. A GPS stack has to hold every token's
        activations, which is why it scores bounded subgraphs instead.
        """

        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        # Validated even though it changes nothing here, so a budget the
        # implicit mixer rejects does not pass silently on this one.
        if token_microbatch_size is not None and (
            isinstance(token_microbatch_size, bool)
            or not isinstance(token_microbatch_size, int)
            or token_microbatch_size < 1
        ):
            raise ValueError("token_microbatch_size must be a positive integer")
        ids = _graph_id_tuple(graph_ids, num_graphs=self.num_graphs)
        return PreparedGPSGraph(ids, self.correction(hidden, ids))

    def prepare_from_chunks(
        self,
        chunks: Iterator[tuple[int, Tensor]],
        *,
        graph_ids: Sequence[int] | Tensor,
        token_count: int,
        token_microbatch_size: int,
        rnf_seed: int | None = None,
        offsets: Sequence[int] | None = None,
    ) -> PreparedGPSGraph:
        """Collect the chunks and run the stack over all of them at once.

        Callers stream a span in token microbatches for the implicit mixer's
        benefit. A GPS stack cannot consume a span piecewise -- its attention
        and normalization both span the whole subgraph -- so the chunks are
        rejoined here. The span is a bounded subgraph, so this is affordable.
        """

        ids = _graph_id_tuple(graph_ids, num_graphs=self.num_graphs)
        collected = []
        expected_start = 0
        for start, chunk in chunks:
            if start != expected_start:
                raise ValueError("mixer chunks must cover the span in order")
            collected.append(chunk.to(device=self.device))
            expected_start = start + chunk.size(1)
            del chunk
        if not collected or expected_start != token_count:
            raise ValueError("mixer chunks do not cover the complete span")
        # The pieces and the span they join into are the two largest tensors in
        # the step, so the pieces are released before the stack runs rather than
        # staying alive beside it. A single piece is already the whole span.
        span = collected[0] if len(collected) == 1 else torch.cat(collected, dim=1)
        collected.clear()
        return self.prepare(
            span, ids, token_microbatch_size=token_microbatch_size
        )

    def correction_from_prepared(
        self, prepared: PreparedGPSGraph
    ) -> Tensor | GateInjection:
        if not isinstance(prepared, PreparedGPSGraph):
            raise ValueError("the GPS mixer requires prepared GPS state")
        return prepared.correction

    def delta_from_prepared(self, prepared: PreparedGPSGraph) -> Tensor:
        correction = self.correction_from_prepared(prepared)
        if not isinstance(correction, Tensor):
            raise ValueError("the gate-space coupling produces an injection, not a delta")
        return correction

    def coupling_config(self) -> dict[str, object]:
        """The coupling settings this mixer applies, recorded only when applied."""

        config: dict[str, object] = {"mixer_coupling": self.coupling}
        if self.injection is not None:
            config["injection_target"] = self.injection.target
            config["injection_init"] = self.injection.init_std
        return config

    def forward(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        rnf_seed: int | None = None,
    ) -> Tensor:
        """Convenience path, matching the implicit mixer; scoring uses `prepare`.

        `rnf_seed` is accepted and ignored for the same reason `prepare` accepts
        it: callers forward it without asking which mixer they have.
        """

        return self.correction(hidden, graph_ids)


def _keep_probability(base_logits: Tensor, logits: Tensor, *, dim: int) -> Tensor:
    """The gate's keep probability, without overflowing on a wide logit gap.

    The score is 1 / (1 + sum_s exp(base_s - logit)). Written that way the
    exponential overflows to infinity once the gap passes about 88, the sum
    becomes infinite and the score underflows to exactly zero. The forward pass
    survives that, but the backward multiplies the zero local derivative by the
    infinite one from exp and returns NaN, which the optimizer then writes into
    every parameter. Training run 21477027 died exactly that way.

    Rewriting the sum as a log-sum-exp and the reciprocal as a sigmoid is the
    same function, evaluated through two kernels that are stable at any gap.
    """

    return torch.sigmoid(-torch.logsumexp(base_logits - logits, dim=dim))


def _promoted_add(values: Tensor, addend: Tensor) -> Tensor:
    """Add in the wider of the two dtypes.

    The injection is accumulated in the master dtype, so adding it promotes the
    gate math exactly as adding the hidden-width delta does.
    """

    dtype = torch.promote_types(values.dtype, addend.dtype)
    return values.to(dtype) + addend.to(dtype)


def _inject_queries_keys(
    queries: Tensor, keys: Tensor, injection: GateInjection, *, group_axis: int
) -> tuple[Tensor, Tensor]:
    """Add the injection after the gate's RMSNorms.

    Queries carry a query-group axis the injection lacks, so its query term is
    shared across the groups; keys have no such axis.
    """

    if injection.query is not None:
        queries = _promoted_add(queries, injection.query.unsqueeze(group_axis))
    if injection.key is not None:
        keys = _promoted_add(keys, injection.key)
    return queries, keys


def _injection_matches(
    injection: GateInjection, graph_count: int, token_count: int
) -> bool:
    return all(
        value is None
        or (
            value.ndim == 3
            and value.size(0) == graph_count
            and value.size(1) == token_count
        )
        for value in (injection.query, injection.key, injection.logit)
    )


class _HeadwiseGateAdapter(nn.Module):
    """Apply a head-specific mixer correction to matching gate slices.

    The hidden coupling's correction is a hidden-width delta added to the gate
    input. The gate-space coupling's correction is a `GateInjection` added to
    the normalized queries and keys and to the logit bias.
    """

    def forward(
        self,
        gate: nn.Module,
        head: int,
        hidden: Tensor,
        delta: Tensor | None = None,
        injection: GateInjection | None = None,
    ) -> Tensor:
        """Score one head.

        Production scoring calls `forward_batch`; this stays as the readable
        single-head statement of the same math, and the tests use it as the
        oracle `forward_batch` is checked against. Both must therefore accept
        the same inputs: a gate-only scorer's absent correction, the hidden
        coupling's delta, and the gate-space coupling's injection, whose
        tensors here carry no graph axis (`[tokens, ...]`).
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
        if injection is not None:
            queries, keys = _inject_queries_keys(queries, keys, injection, group_axis=1)

        logits = torch.einsum("tr,tgr->tg", keys.to(queries.dtype), queries) / gate.d
        logits = logits + gate.b[head, 0].to(logits.dtype)
        if injection is not None and injection.logit is not None:
            logits = _promoted_add(logits, injection.logit)
        base_logits = torch.einsum(
            "sr,tgr->tsg", gate.k_base[head, 0].to(queries.dtype), queries
        ) / gate.d
        scores = _keep_probability(base_logits, logits.unsqueeze(1), dim=1)
        return scores.mean(dim=-1)

    def forward_batch(
        self,
        gates: Sequence[nn.Module],
        layer_ids: Sequence[int],
        head_ids: Sequence[int],
        hidden: Tensor,
        correction: "Tensor | GateInjection | None" = None,
    ) -> Tensor:
        """Apply matching gate heads to a complete graph microbatch.

        A gate-only scorer passes no correction; the gate then reads the hidden
        states unchanged instead of adding a zero tensor of their size. A
        hidden-width tensor is the hidden coupling's delta; a `GateInjection`
        is the gate-space coupling's addition to queries, keys and logits.
        """

        layer_ids = tuple(layer_ids)
        head_ids = tuple(head_ids)
        graph_count = len(layer_ids)
        delta = correction if isinstance(correction, Tensor) else None
        injection = correction if isinstance(correction, GateInjection) else None
        if correction is not None and delta is None and injection is None:
            raise ValueError("correction must be a hidden-width delta or a gate injection")
        if (
            not graph_count
            or hidden.ndim != 3
            or (delta is not None and delta.shape != hidden.shape)
            or (
                injection is not None
                and not _injection_matches(injection, graph_count, hidden.size(1))
            )
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
                # An RMSNorm whose weight is wider than its input returns the
                # wider dtype, so the rows written back can outrank the tensor
                # they are written into. Widen the whole tensor rather than
                # narrowing the rows: the single-head path keeps that
                # precision, and these two must agree.
                if normalized.dtype != result.dtype:
                    result = result.to(torch.promote_types(result.dtype, normalized.dtype))
                    normalized = normalized.to(result.dtype)
                result = result.index_copy(0, positions, normalized)
            return result

        queries = normalize(queries, "q_norm")
        keys = normalize(keys, "k_norm")
        if injection is not None:
            queries, keys = _inject_queries_keys(queries, keys, injection, group_axis=2)

        logits = torch.einsum("mtr,mtgr->mtg", keys.to(queries.dtype), queries) / first_gate.d
        logits = logits + torch.stack(
            [gate.b[head, 0] for gate, head in selected]
        ).to(logits.dtype).unsqueeze(1)
        if injection is not None and injection.logit is not None:
            logits = _promoted_add(logits, injection.logit)
        k_base = torch.stack(
            [gate.k_base[head, 0] for gate, head in selected]
        ).to(queries.dtype)
        base_logits = torch.einsum("msr,mtgr->mtsg", k_base, queries) / first_gate.d
        scores = _keep_probability(base_logits, logits.unsqueeze(2), dim=2)
        return scores.mean(dim=-1)


class ImplicitGraphScorer(nn.Module):
    """Score one context with independent implicit mixers per layer and KV head."""

    def __init__(
        self,
        gates,
        model_config,
        *,
        graph_dim: int | None = 32,
        mixer_architecture: str = DEFAULT_MIXER_ARCHITECTURE,
        gps_depth: int = 1,
        gps_attention_heads: int = GPS_DEFAULT_ATTENTION_HEADS,
        gps_random_features: int = GPS_DEFAULT_RANDOM_FEATURES,
        gps_redraw_interval: int = 0,
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
        mixer_coupling: str = DEFAULT_MIXER_COUPLING,
        injection_target: str = DEFAULT_INJECTION_TARGET,
        injection_init: float = DEFAULT_INJECTION_INIT,
        self_loop_init: float | None = None,
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
        self.mixer_architecture = parse_mixer_architecture(mixer_architecture)
        self.mixer_coupling = parse_mixer_coupling(mixer_coupling)
        if self_loop_init is not None and self.mixer_architecture != "implicit":
            raise ValueError(
                "self loops apply to the implicit mixer; a gps block already has a "
                "residual around its aggregation"
            )
        if graph_dim is None:
            # A gate-only scorer has no mixer, so the architecture is moot.
            self.mixer = None
        elif self.mixer_architecture == "implicit":
            self.mixer = ImplicitGraphMixer(
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
                coupling=self.mixer_coupling,
                injection_target=injection_target,
                injection_init=injection_init,
                self_loop_init=self_loop_init,
                gate_dim=self.gate_dim,
                query_groups=first_gate.ngroup,
                device=device,
                dtype=master_dtype,
            )
        else:
            self.mixer = GPSGraphMixer(
                self.num_graphs,
                self.hidden_dim,
                graph_dim,
                depth=gps_depth,
                attention_heads=gps_attention_heads,
                random_features=gps_random_features,
                redraw_interval=gps_redraw_interval,
                gram_normalization=gram_normalization,
                leaky_relu_slope=leaky_relu_slope,
                alpha_init=alpha_init,
                coupling=self.mixer_coupling,
                injection_target=injection_target,
                injection_init=injection_init,
                gate_dim=self.gate_dim,
                query_groups=first_gate.ngroup,
                device=device,
                dtype=master_dtype,
            )
        self._gate_adapter = _HeadwiseGateAdapter()

    @property
    def scores_subgraphs_only(self) -> bool:
        """Whether this scorer refuses whole-context scoring.

        A GPS stack keeps every token's activations, so it is trained and
        evaluated on bounded subgraphs rather than a whole context.
        """

        return isinstance(self.mixer, GPSGraphMixer)

    def _require_whole_context(self) -> None:
        if self.scores_subgraphs_only:
            raise ValueError(
                "the gps mixer scores fixed-size subgraphs; set a subgraph size "
                "instead of scoring a whole context"
            )

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

        With the hidden coupling the delta is accumulated in the master dtype,
        so adding it promotes the gate input to that dtype regardless of this
        value. A gate-only scorer has no delta to promote it, so it
        materializes the hidden states in the master dtype directly; otherwise
        dropping the mixer would silently drop the gate to a lower precision,
        confounding the ablation with a dtype change. The gate-space coupling
        adds nothing to the gate input either, so it follows the gate-only
        rule and the gate's projections run at the same precision under both
        couplings.
        """

        if self.mixer is not None and self.mixer.coupling == "hidden":
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
    ) -> PreparedImplicitGraph | PreparedGPSGraph | None:
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
        prepared: PreparedImplicitGraph | PreparedGPSGraph | None,
        *,
        layer_ids: Sequence[int] | Tensor,
        head_ids: Sequence[int] | Tensor,
    ) -> tuple[Tensor, Tensor | GateInjection | None]:
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
        correction = (
            None if prepared is None else self.mixer.correction_from_prepared(prepared)
        )
        scores = self._gate_adapter.forward_batch(
            self.gates,
            layer_ids,
            head_ids,
            hidden.to(device=self.device, dtype=self.hidden_dtype),
            correction,
        )
        return scores, correction

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
        self._require_whole_context()
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
