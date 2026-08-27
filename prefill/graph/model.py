"""Per-layer, per-KV-head graph scorer."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric import EdgeIndex

from .builder import FaissGraphBuilder, GraphBuilder


def resolve_graph_microbatch_size(value: str | int, layers: int, heads: int) -> int:
    """Resolve ``auto`` or validate a complete-graph microbatch size."""

    graph_count = layers * heads
    if value == "auto":
        return heads
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= graph_count:
        raise ValueError(f"graph microbatch size must be an integer from 1 to {graph_count}")
    return value


class PerGraphLinear(nn.Module):
    """Bias-free linear projections with one weight per flattened graph."""

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

    def forward(self, x: Tensor, graph_ids: Tensor) -> Tensor:
        return torch.einsum("bti,boi->bto", x, self.weight[graph_ids])


class GroupedGIN(nn.Module):
    """GIN with independent epsilon and MLP parameters for every graph."""

    def __init__(
        self,
        num_graphs: int,
        dim: int,
        depth: int = 1,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_graphs < 1 or dim < 1 or depth < 1:
            raise ValueError("num_graphs, dim, and depth must be positive")
        self.num_graphs = num_graphs
        self.dim = dim
        self.depth = depth
        self.eps = nn.Parameter(torch.zeros(depth, num_graphs, device=device, dtype=dtype))
        self.mlps = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(dim, dim, device=device, dtype=dtype),
                            nn.ReLU(),
                            nn.Linear(dim, dim, device=device, dtype=dtype),
                        )
                        for _ in range(num_graphs)
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        z: Tensor,
        edge_index: EdgeIndex,
        graph_ids: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if z.ndim != 3 or graph_ids.ndim != 1 or z.size(0) != graph_ids.numel():
            raise ValueError("z and graph_ids must have shapes [B,T,R] and [B]")
        if z.size(-1) != self.dim:
            raise ValueError(f"expected node dimension {self.dim}, got {z.size(-1)}")
        graph_ids = graph_ids.to(device=self.eps.device, dtype=torch.long)
        if torch.any(graph_ids < 0) or torch.any(graph_ids >= self.num_graphs):
            raise ValueError("graph_ids contains an unknown graph")

        batch_size, token_count, _ = z.shape
        x = z.reshape(batch_size * token_count, self.dim)
        source, target = edge_index
        for layer, layer_mlps in enumerate(self.mlps):
            messages = x[source]
            if edge_weight is not None:
                messages = messages * edge_weight.unsqueeze(-1)
            aggregate = torch.zeros_like(x).index_add(0, target, messages)
            updated = []
            for local_graph, graph_id_tensor in enumerate(graph_ids):
                graph_id = int(graph_id_tensor)
                start = local_graph * token_count
                stop = start + token_count
                combined = (
                    (1 + self.eps[layer, graph_id]) * x[start:stop]
                    + aggregate[start:stop]
                )
                updated.append(layer_mlps[graph_id](combined))
            x = torch.cat(updated, dim=0)
        return x.view(batch_size, token_count, self.dim)


class _HeadwiseGateAdapter(nn.Module):
    """Apply a head-specific hidden-state delta to matching gate slices."""

    def forward(self, gate: nn.Module, head: int, hidden: Tensor, delta: Tensor) -> Tensor:
        token_count = hidden.size(0)
        gate_dim = gate.output_dim
        groups = gate.ngroup
        mixed = hidden + delta

        q_weight = gate.q_proj.weight.view(
            gate.nhead, groups * gate_dim, gate.q_proj.in_features
        )[head]
        q_bias = None
        if gate.q_proj.bias is not None:
            q_bias = gate.q_proj.bias.view(gate.nhead, groups * gate_dim)[head]
        queries = F.linear(mixed, q_weight, q_bias)
        queries = gate.q_norm(queries.view(token_count, groups, gate_dim))

        k_weight = gate.k_proj.weight.view(
            gate.nhead, gate_dim, gate.k_proj.in_features
        )[head]
        keys = gate.k_norm(F.linear(mixed, k_weight))

        logits = torch.einsum("tr,tgr->tg", keys, queries) / gate.d
        logits = logits + gate.b[head, 0]
        base_logits = torch.einsum("sr,tgr->tsg", gate.k_base[head, 0], queries) / gate.d
        scores = 1 / (1 + torch.exp(base_logits - logits.unsqueeze(1)).sum(dim=1))
        return scores.mean(dim=-1)


class GraphScorer(nn.Module):
    """Score one whole context with a graph per layer and KV head."""

    def __init__(
        self,
        gates,
        model_config,
        *,
        graph_dim: int = 32,
        gin_depth: int = 1,
        graph_builder: GraphBuilder | None = None,
        graph_microbatch_size: str | int = "auto",
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
        self.hidden_dim = first_gate.q_proj.in_features
        self.gate_dim = first_gate.output_dim
        if any(
            gate.nhead != self.num_heads
            or gate.q_proj.in_features != self.hidden_dim
            or gate.output_dim != self.gate_dim
            for gate in self.gates
        ):
            raise ValueError("runtime gate dimensions do not match model configuration")

        resolve_graph_microbatch_size(
            graph_microbatch_size, self.num_layers, self.num_heads
        )
        self.graph_microbatch_size = graph_microbatch_size
        factory_kwargs = {
            "device": first_gate.q_proj.weight.device,
            "dtype": first_gate.q_proj.weight.dtype,
        }
        self.a_proj = PerGraphLinear(
            self.num_graphs, self.hidden_dim, graph_dim, **factory_kwargs
        )
        self.gin = GroupedGIN(
            self.num_graphs, graph_dim, gin_depth, **factory_kwargs
        )
        self.b_proj = PerGraphLinear(
            self.num_graphs, graph_dim, self.hidden_dim, **factory_kwargs
        )
        nn.init.zeros_(self.b_proj.weight)
        self.graph_builder = graph_builder or FaissGraphBuilder()
        self._gate_adapter = _HeadwiseGateAdapter()
        self.register_buffer(
            "_last_delta_energy_share",
            torch.zeros((), **factory_kwargs),
            persistent=False,
        )

    @property
    def last_delta_energy_share(self) -> Tensor:
        return self._last_delta_energy_share

    def graph_batches(
        self,
        *,
        microbatch_size: str | int | None = None,
        device=None,
    ):
        size = resolve_graph_microbatch_size(
            self.graph_microbatch_size if microbatch_size is None else microbatch_size,
            self.num_layers,
            self.num_heads,
        )
        device = self.a_proj.weight.device if device is None else device
        for start in range(0, self.num_graphs, size):
            graph_ids = torch.arange(
                start, min(start + size, self.num_graphs), device=device
            )
            yield (
                graph_ids,
                torch.div(graph_ids, self.num_heads, rounding_mode="floor"),
                graph_ids.remainder(self.num_heads),
            )

    def project_graph_nodes(self, graph_hidden: Tensor, graph_ids: Tensor) -> Tensor:
        return self.a_proj(graph_hidden, graph_ids)

    def propagate_graph_nodes(self, z: Tensor, graph_ids: Tensor) -> Tensor:
        topology = self.graph_builder(z)
        return self.gin(z, topology.edge_index, graph_ids, topology.edge_weight)

    def score_mixed_graph_nodes(
        self,
        graph_hidden: Tensor,
        u: Tensor,
        graph_ids: Tensor,
        layer_ids: Tensor,
        head_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        delta = self.b_proj(u, graph_ids)
        scores = torch.stack(
            [
                self._gate_adapter(
                    self.gates[int(layer_id)],
                    int(head_id),
                    graph_hidden[local_graph],
                    delta[local_graph],
                )
                for local_graph, (layer_id, head_id) in enumerate(
                    zip(layer_ids, head_ids)
                )
            ]
        )
        return scores, delta

    def forward(
        self, hidden: Tensor, *, microbatch_size: str | int | None = None
    ) -> Tensor:
        if hidden.ndim == 4 and hidden.size(1) == 1:
            hidden = hidden[:, 0]
        if hidden.ndim != 3 or hidden.size(0) != self.num_layers:
            raise ValueError("hidden must have shape [layers,T,D] or [layers,1,T,D]")
        if hidden.size(-1) != self.hidden_dim:
            raise ValueError(f"expected hidden dimension {self.hidden_dim}")
        score_batches = []
        delta_energy = hidden.new_zeros(())
        hidden_energy = hidden.new_zeros(())
        for graph_ids, layer_ids, head_ids in self.graph_batches(
            microbatch_size=microbatch_size,
            device=hidden.device,
        ):
            graph_hidden = hidden[layer_ids]
            z = self.project_graph_nodes(graph_hidden, graph_ids)
            u = self.propagate_graph_nodes(z, graph_ids)
            scores, delta = self.score_mixed_graph_nodes(
                graph_hidden, u, graph_ids, layer_ids, head_ids
            )
            score_batches.append(scores)
            delta_energy = delta_energy + delta.square().sum()
            hidden_energy = hidden_energy + graph_hidden.square().sum()

        denominator = delta_energy + hidden_energy
        share = torch.where(denominator > 0, delta_energy / denominator, denominator)
        self._last_delta_energy_share = share.detach()
        flat_scores = torch.cat(score_batches, dim=0)
        return flat_scores.view(
            self.num_layers, self.num_heads, hidden.size(1)
        ).unsqueeze(1)
