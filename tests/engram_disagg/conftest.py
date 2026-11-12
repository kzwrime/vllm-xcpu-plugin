from types import SimpleNamespace

import pytest

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiCluster, MpiWorld


def make_world(
    cluster_type=ClusterType.ATTN,
    *,
    num_attention_ranks=2,
    num_engram_ranks=2,
    role_rank=0,
    extra_clusters=(),
):
    """CPU-only MPMD world; real MPI is covered by torch_xcpu's round-trip test."""
    attention_ranks = tuple(range(num_attention_ranks))
    engram_ranks = tuple(
        range(num_attention_ranks, num_attention_ranks + num_engram_ranks)
    )
    role_ranks = attention_ranks if cluster_type == ClusterType.ATTN else engram_ranks
    clusters = {
        0: MpiCluster(0, ClusterType.ATTN, attention_ranks),
        2: MpiCluster(2, ClusterType.ENGRAM, engram_ranks),
    }
    for cluster in extra_clusters:
        clusters[cluster.instance_id] = cluster
    world_size = max(r for c in clusters.values() for r in c.global_ranks) + 1
    global_rank = role_ranks[role_rank]
    pair_ranks = sorted(attention_ranks + engram_ranks)
    global_comm = SimpleNamespace(
        Get_rank=lambda: global_rank, Get_size=lambda: world_size, py2f=lambda: 7
    )
    # The attention/Engram pair: ranks ordered by global rank.
    global_comm.Split = lambda color, key: SimpleNamespace(
        Get_rank=lambda: pair_ranks.index(global_rank),
        Get_size=lambda: len(pair_ranks),
        py2f=lambda: 9,
    )
    return MpiWorld(
        cluster_instance_id=int(cluster_type),
        cluster_type=cluster_type,
        cluster_comm=SimpleNamespace(
            Get_rank=lambda: role_rank, Get_size=lambda: len(role_ranks)
        ),
        global_world_comm=global_comm,
        clusters=clusters,
    )


@pytest.fixture
def world():
    return make_world


@pytest.fixture
def fake_remote_ops(monkeypatch):
    """Replace the native endpoints with recorders; initialize returns the
    attention proposal as the negotiated capacity."""
    import torch
    import torch_xcpu

    calls = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: calls.append("sync"))

    def initialize(metadata, comm, max_tokens, layers, total_cols, cols_per_rank, nums):
        calls.append((
            "initialize",
            max_tokens,
            layers,
            total_cols,
            cols_per_rank,
            nums,
        ))
        return max_tokens or 16

    monkeypatch.setattr(
        torch_xcpu.ops, "deepseek_v41_engram_remote_initialize", initialize
    )
    for name in ("request", "collect", "decode", "stop", "cleanup"):
        monkeypatch.setattr(
            torch_xcpu.ops,
            f"deepseek_v41_engram_remote_{name}",
            lambda *args, _name=name: calls.append((_name, *args)),
        )
    return calls
