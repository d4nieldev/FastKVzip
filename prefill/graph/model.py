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
GPS_ACTIVATION_ORDER = "layernorm-gelu-leaky-relu"

MIXER_ARCHITECTURES = ("implicit", "gps")
DEFAULT_MIXER_ARCHITECTURE = "implicit"

GPS_DEFAULT_ATTENTION_HEADS = 4
GPS_DEFAULT_RANDOM_FEATURES = 32
# Fixed rather than exposed: one more knob per architecture buys little next to
# graph width, which already controls the block's size.
GPS_FFN_MULTIPLIER = 2


def parse_mixer_architecture(value: object) -> str:
    if value not in MIXER_ARCHITECTURES:
        raise ValueError(
            f"mixer architecture must be one of {', '.join(MIXER_ARCHITECTURES)}"
        )
    return str(value)


def mixer_activation_order(architecture: str) -> str:
    """Activation order recorded for, and validated against, one architecture."""

    return ACTIVATION_ORDER if parse_mixer_architecture(architecture) == "implicit" else GPS_ACTIVATION_ORDER


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

    def select_tokens(self, index: Tensor) -> "PreparedImplicitGraph":
        """Restrict to some tokens, keeping the whole context's statistics."""

        return PreparedImplicitGraph(
            self.graph_ids,
            self.y1.index_select(1, index.to(self.y1.device)),
            self.gram,
            self.kernel,
            self.norm,
            self.token_count,
        )

    def narrow_tokens(self, start: int, length: int) -> "PreparedImplicitGraph":
        """Restrict to a contiguous run of tokens, as a view."""

        return PreparedImplicitGraph(
            self.graph_ids,
            self.y1.narrow(1, start, length),
            self.gram,
            self.kernel,
            self.norm,
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

    def parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        """Split parameters into weight-decayed and undecayed groups."""

        return [self.in_proj.weight, self.out_proj.weight], [self.alpha, self.gamma, self.beta]

    def on_optimizer_step(self) -> None:
        """Nothing here is resampled during training."""

    def delta_from_prepared(self, prepared: PreparedImplicitGraph) -> Tensor:
        if not isinstance(prepared, PreparedImplicitGraph):
            raise ValueError("the implicit mixer requires prepared implicit state")
        return self.delta(prepared.y1, prepared)

    def forward(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        """Convenience full-output path for small tests; scoring uses streamed APIs."""

        prepared = self.prepare(
            hidden, graph_ids, token_microbatch_size=max(1, hidden.size(1))
        )
        return self.delta(prepared.y1, prepared)


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
    """A GPS stack's finished hidden-state delta for one token span."""

    graph_ids: tuple[int, ...]
    delta: Tensor

    def select_tokens(self, index: Tensor) -> "PreparedGPSGraph":
        # Selecting every token in order is what validation always asks for, and
        # index_select would copy the whole correction to answer it.
        if index.numel() == self.delta.size(1) and bool(
            torch.equal(index.cpu(), torch.arange(index.numel()))
        ):
            return self
        return PreparedGPSGraph(
            self.graph_ids, self.delta.index_select(1, index.to(self.delta.device))
        )

    def narrow_tokens(self, start: int, length: int) -> "PreparedGPSGraph":
        """Restrict to a contiguous run of tokens, as a view."""

        return PreparedGPSGraph(self.graph_ids, self.delta.narrow(1, start, length))


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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or hidden_dim < 1 or graph_dim < 1:
            raise ValueError("num_graphs, hidden_dim, and graph_dim must be positive")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("gps depth must be a positive integer")
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
        self.gram_normalization = gram_normalization
        self.leaky_relu_slope = float(leaky_relu_slope)
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
        self.out_proj = PerGraphLinear(
            num_graphs, graph_dim, hidden_dim, device=device, dtype=dtype
        )
        self.alpha = nn.Parameter(torch.full((num_graphs,), alpha_init, device=device, dtype=dtype))
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

    def parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        """Split parameters into weight-decayed and undecayed groups."""

        no_decay = [self.alpha]
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

    def delta(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [graphs,tokens,hidden_dim]")
        x = self.in_proj(hidden, graph_ids)
        x = x + sinusoidal_positions(
            x.size(1), self.graph_dim, device=x.device, dtype=x.dtype
        )
        for block in self.blocks:
            x = block(x, graph_ids)
        projected = self.out_proj(x, graph_ids)
        # Return the delta in the reduction dtype, as the implicit mixer does.
        # Adding it is what promotes the gate's input above the compute dtype,
        # and the gate's own normalization returns that higher precision either
        # way; a delta left in bfloat16 makes the two disagree.
        dtype = _reduction_dtype(projected)
        alpha = _select_graph_rows(self.alpha, graph_ids).to(dtype).view(-1, 1, 1)
        return alpha * F.leaky_relu(
            projected.to(dtype), negative_slope=self.leaky_relu_slope
        )

    def prepare(
        self,
        hidden: Tensor,
        graph_ids: Sequence[int] | Tensor,
        *,
        token_microbatch_size: int | None = None,
    ) -> PreparedGPSGraph:
        """Run the stack once. Token microbatching does not apply to GPS.

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
        return PreparedGPSGraph(ids, self.delta(hidden, ids))

    def prepare_from_chunks(
        self,
        chunks: Iterator[tuple[int, Tensor]],
        *,
        graph_ids: Sequence[int] | Tensor,
        token_count: int,
        token_microbatch_size: int,
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
        for start, hidden in chunks:
            if start != expected_start:
                raise ValueError("mixer chunks must cover the span in order")
            collected.append(hidden.to(device=self.device))
            expected_start = start + hidden.size(1)
        if not collected or expected_start != token_count:
            raise ValueError("mixer chunks do not cover the complete span")
        return self.prepare(torch.cat(collected, dim=1), ids)

    def delta_from_prepared(self, prepared: PreparedGPSGraph) -> Tensor:
        if not isinstance(prepared, PreparedGPSGraph):
            raise ValueError("the GPS mixer requires prepared GPS state")
        return prepared.delta

    def forward(self, hidden: Tensor, graph_ids: Sequence[int] | Tensor) -> Tensor:
        """Convenience path, matching the implicit mixer; scoring uses `prepare`."""

        return self.delta(hidden, graph_ids)


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
        mixer_architecture: str = DEFAULT_MIXER_ARCHITECTURE,
        gps_depth: int = 1,
        gps_attention_heads: int = GPS_DEFAULT_ATTENTION_HEADS,
        gps_random_features: int = GPS_DEFAULT_RANDOM_FEATURES,
        gps_redraw_interval: int = 0,
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
        self.mixer_architecture = parse_mixer_architecture(mixer_architecture)
        if graph_dim is None:
            # A gate-only scorer has no mixer, so the architecture is moot.
            self.mixer = None
        elif self.mixer_architecture == "implicit":
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
    ) -> PreparedImplicitGraph | PreparedGPSGraph | None:
        if self.mixer is None:
            return None
        hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
        return self.mixer.prepare(
            hidden, graph_ids, token_microbatch_size=token_microbatch_size
        )

    def score_prepared(
        self,
        hidden: Tensor,
        prepared: PreparedImplicitGraph | PreparedGPSGraph | None,
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
        delta = None if prepared is None else self.mixer.delta_from_prepared(prepared)
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
            hidden, batch.graph_ids * count, token_microbatch_size=length
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
    ) -> Tensor:
        self._require_whole_context()
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

            prepared = (
                None
                if self.mixer is None
                else self.mixer.prepare_from_chunks(
                    chunks(),
                    graph_ids=batch.graph_ids,
                    token_count=token_count,
                    token_microbatch_size=token_microbatch_size,
                )
            )
            chunks_scores = []
            for start in range(0, token_count, token_microbatch_size):
                stop = min(start + token_microbatch_size, token_count)
                graph_hidden = torch.stack(
                    tuple(hidden[layer_id, start:stop] for layer_id in batch.layer_ids)
                ).to(device=self.device, dtype=self.hidden_dtype)
                slice_prepared = (
                    None if prepared is None else prepared.narrow_tokens(start, stop - start)
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
