from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.common.session_v7 import AfV7Session
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiCluster, MpiWorld


@pytest.fixture
def make_af_session(monkeypatch):
    """CPU-only AF setup; real MPI is covered by torch_xcpu's round-trip test."""
    import torch_xcpu

    monkeypatch.setenv("VLLM_XCPU_AF_FORWARD_ALLREDUCE", "0")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch_xcpu.ops, "moe_af_v7_initialize", lambda *args: None)

    def make(
        cluster_type=ClusterType.ATTN,
        max_rows=8,
        num_attention_ranks=2,
        num_expert_ranks=2,
        role_rank=0,
    ):
        instance = int(cluster_type)
        attention_ranks = tuple(range(num_attention_ranks))
        expert_ranks = tuple(
            range(num_attention_ranks, num_attention_ranks + num_expert_ranks)
        )
        role_ranks = (
            attention_ranks if cluster_type == ClusterType.ATTN else expert_ranks
        )
        world = MpiWorld(
            cluster_instance_id=instance,
            cluster_type=cluster_type,
            cluster_comm=SimpleNamespace(
                Get_rank=lambda: role_rank,
                Get_size=lambda: len(role_ranks),
            ),
            global_world_comm=SimpleNamespace(
                Get_rank=lambda: role_ranks[role_rank],
                Get_size=lambda: num_attention_ranks + num_expert_ranks,
                py2f=lambda: 7,
            ),
            clusters={
                0: MpiCluster(0, ClusterType.ATTN, attention_ranks),
                1: MpiCluster(1, ClusterType.MOE, expert_ranks),
            },
        )
        return AfV7Session(world, max_rows_per_attention_rank=max_rows)

    return make
