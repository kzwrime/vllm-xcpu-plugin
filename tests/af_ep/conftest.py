from types import SimpleNamespace

import pytest
import torch

from vllm_xcpu_plugin.af_ep.common.session_v7 import AfV7Session
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, MpiCluster, MpiWorld


@pytest.fixture
def make_af_session(monkeypatch):
    """CPU-only A2/F2 setup; real MPI is covered by torch_xcpu's round-trip test."""
    import torch_xcpu

    monkeypatch.setenv("VLLM_XCPU_AF_FORWARD_ALLREDUCE", "0")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch_xcpu.ops, "moe_af_v7_initialize", lambda *args: None)

    def make(cluster_type=ClusterType.ATTN, max_rows=8):
        instance = int(cluster_type)
        world = MpiWorld(
            cluster_instance_id=instance,
            cluster_type=cluster_type,
            cluster_comm=SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 2),
            global_world_comm=SimpleNamespace(
                Get_rank=lambda: instance * 2, Get_size=lambda: 4, py2f=lambda: 7
            ),
            clusters={
                0: MpiCluster(0, ClusterType.ATTN, (0, 1)),
                1: MpiCluster(1, ClusterType.MOE, (2, 3)),
            },
        )
        return AfV7Session(world, max_rows_per_attention_rank=max_rows)

    return make
