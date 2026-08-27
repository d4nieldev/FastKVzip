import faiss
import pytest
import torch
from torch_geometric import EdgeIndex

import graph.builder as builder_module


def _builder_class():
    cls = getattr(builder_module, "FaissGraphBuilder", None)
    assert cls is not None, "FaissGraphBuilder is not implemented"
    return cls


def test_exact_fallback_builds_directed_neighbor_to_token_edges_without_self_edges():
    graph_builder = _builder_class()(nlist=8)
    z = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]],
        requires_grad=True,
    )

    topology = graph_builder(z, 1)

    assert isinstance(topology.edge_index, EdgeIndex)
    assert topology.edge_weight is None
    assert set(map(tuple, topology.edge_index.t().tolist())) == {
        (1, 0),
        (0, 1),
        (3, 2),
        (2, 3),
    }
    assert torch.equal(
        torch.bincount(topology.edge_index[1], minlength=4),
        torch.ones(4, dtype=torch.long),
    )
    assert not torch.any(topology.edge_index[0] == topology.edge_index[1])
    assert z.requires_grad


def test_k_is_capped_at_other_token_count():
    graph_builder = _builder_class()(nlist=8)
    topology = graph_builder(torch.eye(3).unsqueeze(0), 9)

    assert torch.equal(
        torch.bincount(topology.edge_index[1], minlength=3),
        torch.full((3,), 2, dtype=torch.long),
    )
    assert not torch.any(topology.edge_index[0] == topology.edge_index[1])


@pytest.mark.parametrize(
    ("index_mode", "constructor_name"),
    [("ivf_flat", "IndexIVFFlat"), ("ivf_pq", "IndexIVFPQ")],
)
def test_requested_approximate_faiss_index_mode_is_used(
    monkeypatch, index_mode, constructor_name
):
    calls = []
    real_constructor = getattr(faiss, constructor_name)

    def recording_constructor(*args):
        calls.append(args)
        return real_constructor(*args)

    monkeypatch.setattr(faiss, constructor_name, recording_constructor)
    graph_builder = _builder_class()(
        index_mode=index_mode,
        nlist=2,
        nprobe=2,
        pq_bits=2,
    )
    z = torch.randn(32, 4, generator=torch.Generator().manual_seed(7)).unsqueeze(0)

    topology = graph_builder(z, 2)

    assert len(calls) == 1
    assert torch.equal(
        torch.bincount(topology.edge_index[1], minlength=32),
        torch.full((32,), 2, dtype=torch.long),
    )


@pytest.mark.parametrize("index_mode", ["ivf_flat", "ivf_pq"])
def test_short_context_uses_exact_fallback(monkeypatch, index_mode):
    def approximate_index_must_not_be_created(*_args):
        raise AssertionError("short contexts must use exact search")

    monkeypatch.setattr(faiss, "IndexIVFFlat", approximate_index_must_not_be_created)
    monkeypatch.setattr(faiss, "IndexIVFPQ", approximate_index_must_not_be_created)
    graph_builder = _builder_class()(
        index_mode=index_mode,
        nlist=8,
        nprobe=2,
        pq_bits=2,
    )

    topology = graph_builder(torch.randn(4, 4).unsqueeze(0), 1)

    assert topology.edge_index.size(1) == 4


@pytest.mark.parametrize(
    ("graph_dim", "expected"), [(32, 8), (12, 6), (10, 5), (9, 3), (7, 7)]
)
def test_auto_pq_m_is_largest_divisor_not_exceeding_eight(graph_dim, expected):
    assert _builder_class().auto_pq_m(graph_dim) == expected


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_forward_rejects_invalid_requested_neighbor_count(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        _builder_class()(nlist=8)(torch.eye(3).unsqueeze(0), k)
