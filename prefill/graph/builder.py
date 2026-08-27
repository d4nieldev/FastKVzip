"""Graph topology builders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric import EdgeIndex


@dataclass(frozen=True)
class GraphTopology:
    edge_index: EdgeIndex
    edge_weight: Tensor | None = None


class GraphBuilder(nn.Module, ABC):
    """Base class for hard or learnable graph topology builders."""

    @abstractmethod
    def forward(self, z: Tensor) -> GraphTopology:
        raise NotImplementedError


class FaissGraphBuilder(GraphBuilder):
    """Build per-graph directed MIPS neighborhoods with FAISS."""

    def __init__(
        self,
        k: int = 16,
        *,
        index_mode: str = "ivf_flat",
        nlist: int = 256,
        nprobe: int = 16,
        pq_m: int | None = None,
        pq_bits: int = 8,
    ) -> None:
        super().__init__()
        if k < 1 or nlist < 1 or nprobe < 1 or pq_bits < 1:
            raise ValueError("k, nlist, nprobe, and pq_bits must be positive")
        if index_mode not in {"ivf_flat", "ivf_pq"}:
            raise ValueError("index_mode must be 'ivf_flat' or 'ivf_pq'")
        if pq_m is not None and pq_m < 1:
            raise ValueError("pq_m must be positive")
        self.k = k
        self.index_mode = index_mode
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits

    @staticmethod
    def auto_pq_m(graph_dim: int) -> int:
        if graph_dim < 1:
            raise ValueError("graph_dim must be positive")
        return next(m for m in range(min(8, graph_dim), 0, -1) if graph_dim % m == 0)

    def _uses_exact_search(self, token_count: int) -> bool:
        minimum_training_size = self.nlist
        if self.index_mode == "ivf_pq":
            minimum_training_size = max(minimum_training_size, 1 << self.pq_bits)
        return token_count < minimum_training_size

    def _make_index(self, values: np.ndarray):
        token_count, graph_dim = values.shape
        if self._uses_exact_search(token_count):
            index = faiss.IndexFlatIP(graph_dim)
        else:
            quantizer = faiss.IndexFlatIP(graph_dim)
            if self.index_mode == "ivf_flat":
                index = faiss.IndexIVFFlat(
                    quantizer, graph_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                pq_m = self.pq_m or self.auto_pq_m(graph_dim)
                if graph_dim % pq_m:
                    raise ValueError("pq_m must divide graph_dim")
                index = faiss.IndexIVFPQ(
                    quantizer,
                    graph_dim,
                    self.nlist,
                    pq_m,
                    self.pq_bits,
                    faiss.METRIC_INNER_PRODUCT,
                )
            index.nprobe = min(self.nprobe, self.nlist)
            index.train(values)
        index.add(values)
        return index

    def _neighbors(self, values: np.ndarray, k_eff: int) -> np.ndarray:
        index = self._make_index(values)
        _, candidates = index.search(values, k_eff + 1)
        neighbors = np.empty((len(values), k_eff), dtype=np.int64)
        for target, row in enumerate(candidates):
            selected = []
            seen = {target}
            for source in row:
                source = int(source)
                if source >= 0 and source not in seen:
                    selected.append(source)
                    seen.add(source)
            if len(selected) < k_eff:
                exact_order = np.argsort(-(values @ values[target]), kind="stable")
                selected.extend(
                    int(source)
                    for source in exact_order
                    if source not in seen
                )
            neighbors[target] = selected[:k_eff]
        return neighbors

    def forward(self, z: Tensor) -> GraphTopology:
        if z.ndim == 2:
            z = z.unsqueeze(0)
        if z.ndim != 3 or z.size(1) < 1:
            raise ValueError("z must have shape [graphs, tokens, graph_dim] with tokens >= 1")

        graph_count, token_count, _ = z.shape
        k_eff = min(self.k, token_count - 1)
        if k_eff == 0:
            edges = torch.empty((2, 0), dtype=torch.long, device=z.device)
        else:
            sources = []
            targets = []
            for local_graph, graph_z in enumerate(z):
                values = (
                    graph_z.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .numpy()
                )
                neighbors = self._neighbors(values, k_eff)
                offset = local_graph * token_count
                sources.append(torch.from_numpy(neighbors.reshape(-1)) + offset)
                targets.append(torch.arange(token_count).repeat_interleave(k_eff) + offset)
            edges = torch.stack([torch.cat(sources), torch.cat(targets)]).to(z.device)

        edge_index = EdgeIndex(
            edges,
            sparse_size=(graph_count * token_count, graph_count * token_count),
        )
        return GraphTopology(edge_index=edge_index)
