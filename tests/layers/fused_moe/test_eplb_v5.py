# SPDX-License-Identifier: Apache-2.0
"""Unit tests for XCPU EPLB configuration, router rebind, weights, and MPI comm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import vllm.model_executor.layers.fused_moe.router.base_router as base_router
from mpi4py import MPI
from torch_xcpu import ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform

import vllm_xcpu_plugin  # noqa: F401
import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator
from vllm_xcpu_plugin.distributed.mpi_eplb_communicator import MpiEplbCommunicator
from vllm_xcpu_plugin.layers.fp8_moe import XcpuFp8MoEMethod
from vllm_xcpu_plugin.layers.fused_moe.grouped_gemm_experts import (
    XcpuGroupedGemmExperts,
)
from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts
from vllm_xcpu_plugin.layers.fused_moe.unquantized_fused_moe_method import (
    XcpuUnquantizedFusedMoEMethod,
)
from vllm_xcpu_plugin.layers.mxfp4_moe import (
    XcpuCompressedTensorsMxfp4MoEMethod,
    XcpuMxfp4MoEMethod,
)

vllm_xcpu_plugin.xcpu_platform_plugin()
vllm_xcpu_plugin.register_ops()


def test_default_eplb_communicator_selection(monkeypatch):
    monkeypatch.setattr(envs_xcpu, "VLLM_CPU_USE_MPI", True)
    assert current_platform.get_default_eplb_communicator() == "mpi"
    pc_mpi = ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(),
    )
    assert pc_mpi.eplb_config.communicator == "mpi"

    monkeypatch.setattr(envs_xcpu, "VLLM_CPU_USE_MPI", False)
    assert current_platform.get_default_eplb_communicator() == "torch_gloo"
    pc_gloo = ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(),
    )
    assert pc_gloo.eplb_config.communicator == "torch_gloo"

    # Explicit override always wins regardless of VLLM_CPU_USE_MPI.
    monkeypatch.setattr(envs_xcpu, "VLLM_CPU_USE_MPI", True)
    pc_explicit = ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(communicator="torch_gloo"),
    )
    assert pc_explicit.eplb_config.communicator == "torch_gloo"


def test_eplb_router_rebind_executes_torch_xcpu_op():
    logical_to_physical_map = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.int64, device="mcpu"
    )
    logical_replica_count = torch.tensor([2, 2], dtype=torch.int64, device="mcpu")
    expert_load_view = torch.zeros(4, dtype=torch.int32, device="mcpu")
    rec_t = torch.tensor(True, dtype=torch.bool, device="mcpu")
    unpadded_t = torch.tensor(2, dtype=torch.int32, device="mcpu")
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, -1]], dtype=torch.int32, device="mcpu")

    mapped = base_router.eplb_map_to_physical_and_record(
        topk_ids=topk_ids,
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        record_enabled=rec_t,
        num_unpadded_tokens=unpadded_t,
    )
    torch.mcpu.synchronize()
    assert mapped.shape == topk_ids.shape
    assert int(expert_load_view.cpu().sum().item()) == 4


def test_supports_eplb_enabled_on_bf16_fp8_mxfp4_and_grouped_gemm():
    moe_cfg = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=4,
        num_logical_experts=6,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            dp_size=2,
            dp_rank=0,
            pcp_size=1,
            pcp_rank=0,
            ep_size=2,
            ep_rank=0,
            sp_size=1,
            use_ep=True,
            enable_eplb=True,
            all2all_backend="mpi_alltoallv_v5",
        ),
        in_dtype=torch.bfloat16,
        device="mcpu",
        routing_method=RoutingMethodType.TopK,
        activation=MoEActivation.SILU,
    )
    assert XcpuGroupedGemmExperts._supports_parallel_config(
        moe_cfg.moe_parallel_config
    )
    v6_parallel_cfg = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        ep_size=2,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        enable_eplb=True,
        all2all_backend="mpi_alltoallv_v6",
    )
    assert XcpuGroupedGemmExperts._supports_parallel_config(v6_parallel_cfg)

    bad_backend_cfg = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        ep_size=2,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        enable_eplb=True,
        all2all_backend="allgather_reducescatter",
    )
    assert not XcpuGroupedGemmExperts._supports_parallel_config(bad_backend_cfg)

    no_ep_eplb_cfg = FusedMoEParallelConfig(
        tp_size=2,
        tp_rank=0,
        dp_size=1,
        dp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        ep_size=1,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        enable_eplb=True,
        all2all_backend="mpi_alltoallv_v5",
    )
    assert not XcpuGroupedGemmExperts._supports_parallel_config(no_ep_eplb_cfg)
    vllm_cfg = VllmConfig()
    vllm_cfg.model_config = SimpleNamespace(quantization_config=None)
    with set_current_vllm_config(vllm_cfg):
        assert XcpuUnquantizedFusedMoEMethod(moe_cfg).supports_eplb is True
        assert XcpuMxfp4MoEMethod(moe_cfg).supports_eplb is True
        assert XcpuCompressedTensorsMxfp4MoEMethod(moe_cfg).supports_eplb is True

        fake_layer = SimpleNamespace(
            moe_config=moe_cfg,
            params_dtype=torch.bfloat16,
            quant_config=None,
            apply_router_weight_on_input=False,
        )
        fp8_method = XcpuFp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]),
            fake_layer,
        )
        assert fp8_method.supports_eplb is True


def test_xcpu_routed_experts_get_expert_weights_uint8_and_padded_storage():
    moe_cfg = FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=32,
        num_local_experts=2,
        num_logical_experts=2,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            dp_size=2,
            dp_rank=0,
            pcp_size=1,
            pcp_rank=0,
            ep_size=2,
            ep_rank=0,
            sp_size=1,
            use_ep=True,
            enable_eplb=True,
            all2all_backend="mpi_alltoallv_v5",
        ),
        in_dtype=torch.bfloat16,
        device="mcpu",
        routing_method=RoutingMethodType.TopK,
        activation=MoEActivation.SILU,
    )
    mgr = ExpertMapManager(
        max_num_batched_tokens=4,
        top_k=2,
        global_num_experts=4,
        num_redundant_experts=2,
        num_expert_group=None,
        moe_parallel_config=moe_cfg.moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=True,
    )
    with set_current_vllm_config(VllmConfig()):
        layer = XcpuRoutedExperts(
            layer_name="moe",
            params_dtype=torch.bfloat16,
            moe_config=moe_cfg,
            quant_config=None,
            expert_map_manager=mgr,
        )
    # Emulate padded storage via PaddingExampleFp8MoeGroupedGemm on w13/w2.
    w13 = torch.empty((2, 64, 256), dtype=torch.float8_e4m3fn, device="mcpu")
    s13 = torch.ones((2, 1, 2), dtype=torch.float32, device="mcpu")
    w2 = torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn, device="mcpu")
    s2 = torch.ones((2, 2, 1), dtype=torch.float32, device="mcpu")
    gemm1 = ops.initialize_moe_grouped_gemm(
        w13, s13, backend=ops.MoeGroupedGemmBackend.PADDING_EXAMPLE, m_capacity=4
    )
    gemm2 = ops.initialize_moe_grouped_gemm(
        w2, s2, backend=ops.MoeGroupedGemmBackend.PADDING_EXAMPLE, m_capacity=4
    )
    layer._xcpu_fused_moe = SimpleNamespace(
        params=SimpleNamespace(gemm1=gemm1, gemm2=gemm2)
    )
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(gemm1.params.packed_weight, requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(gemm2.params.packed_weight, requires_grad=False),
    )

    expert_weights = list(layer.get_expert_weights())
    assert len(expert_weights) >= 2
    for tensor in expert_weights:
        assert tensor.dtype == torch.uint8
        assert tensor.ndim == 2
        assert tensor.shape[0] == layer.local_num_experts
        assert tensor.is_contiguous()
    assert expert_weights[0].nbytes == gemm1.params.packed_weight_storage.nbytes
    assert expert_weights[0].data_ptr() == gemm1.params.packed_weight_storage.data_ptr()


def test_mpi_eplb_communicator_factory_tags_and_async_thread_check(monkeypatch):
    fake_mpi_comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        Get_attr=lambda attr: 2147483647,
    )
    fake_dev_comm = CpuMPICommunicator.__new__(CpuMPICommunicator)
    fake_dev_comm.mpi_group_comm = fake_mpi_comm
    fake_dev_comm.comm_ptr_wrapper = torch.tensor(
        [123], dtype=torch.int64, device="cpu"
    )
    fake_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        device_communicator=fake_dev_comm,
        cpu_group=object(),
        device_group=object(),
    )

    comm = create_eplb_communicator(fake_group, "mpi", [], [])
    assert isinstance(comm, MpiEplbCommunicator)
    assert comm.needs_profile_buffer_reservation is False

    comm.set_transfer_context(None, layer_idx=3)
    t0 = torch.zeros((4,), dtype=torch.uint8, device="mcpu")
    t1 = torch.zeros((8,), dtype=torch.uint8, device="mcpu")
    comm.add_send([t0, t1], dst_rank=1, expert_id=5)
    comm.add_send([t0, t1], dst_rank=1, expert_id=5)
    tags = [tag for _, _, tag in comm._sends]
    assert len(set(tags)) == 4

    # Verify MPI_TAG_UB=32767 handles >256 experts without collisions, and
    # detects wraparound collisions at >=1024 experts.
    narrow_mpi_comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        Get_attr=lambda attr: 32767,
    )
    narrow_dev_comm = CpuMPICommunicator.__new__(CpuMPICommunicator)
    narrow_dev_comm.mpi_group_comm = narrow_mpi_comm
    narrow_dev_comm.comm_ptr_wrapper = fake_dev_comm.comm_ptr_wrapper
    narrow_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        device_communicator=narrow_dev_comm,
        cpu_group=object(),
        device_group=object(),
    )
    narrow_comm = create_eplb_communicator(narrow_group, "mpi", [], [])
    narrow_comm.set_transfer_context(None, layer_idx=7)
    for exp_id in (0, 256, 512, 1023):
        narrow_comm.add_send([t0, t1], dst_rank=1, expert_id=exp_id)
    narrow_tags = [tag for _, _, tag in narrow_comm._sends]
    assert len(set(narrow_tags)) == 8
    assert all(0 <= tag <= 32767 for tag in narrow_tags)
    with pytest.raises(RuntimeError, match="MPI EPLB tag collision"):
        narrow_comm.add_send([t0, t1], dst_rank=1, expert_id=1024)

    # Verify non-contiguous tensor is rejected.
    with pytest.raises(ValueError, match="contiguous"):
        comm.add_recv([t1[::2]], src_rank=1, expert_id=5)

    # Verify async mode refuses execution without MPI_THREAD_MULTIPLE.
    comm.set_stream(torch.Stream(device="mcpu"))
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_SINGLE)
    comm.add_recv([t0], src_rank=1, expert_id=5)
    with pytest.raises(RuntimeError, match="MPI_THREAD_MULTIPLE"):
        comm.execute()


def test_platform_rejects_indivisible_eplb_physical_experts():
    model_cfg = SimpleNamespace(
        is_moe=True,
        enable_sleep_mode=False,
        get_num_experts=lambda: 128,
        dtype=torch.bfloat16,
    )
    vllm_cfg = SimpleNamespace(
        model_config=model_cfg,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=False,
            enable_eplb=True,
            all2all_backend="mpi_alltoallv_v5",
            eplb_config=EPLBConfig(num_redundant_experts=3, communicator="mpi"),
            data_parallel_size=2,
            tensor_parallel_size=2,
            enable_multicast_communication=False,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_fi_allreduce_fusion=False)
        ),
        kv_cache_config=None,
        speculative_config=None,
        load_config=SimpleNamespace(),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    # Ensure McpuPlatform.check_and_update_config rejects physical=131 % 4 != 0.
    with pytest.raises(ValueError, match="physical_experts"):
        current_platform.check_and_update_config(vllm_cfg)

    # Ensure unsupported all2all_backend is rejected for EPLB.
    vllm_cfg.parallel_config.eplb_config = EPLBConfig(
        num_redundant_experts=4, communicator="mpi"
    )
    vllm_cfg.parallel_config.all2all_backend = "allgather_reducescatter"
    with pytest.raises(ValueError, match="mpi_alltoallv_v5 or mpi_alltoallv_v6"):
        current_platform.check_and_update_config(vllm_cfg)


def test_xcpu_routed_experts_loads_redundant_slots_from_fused_3d_weights():
    moe_config = FusedMoEConfig(
        num_experts=10,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size=8,
        num_local_experts=5,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            dp_size=2,
            pcp_size=1,
            ep_size=2,
            tp_rank=0,
            dp_rank=1,
            pcp_rank=0,
            ep_rank=1,
            sp_size=1,
            use_ep=True,
            all2all_backend="mpi_alltoallv_v5",
            enable_eplb=True,
        ),
        in_dtype=torch.bfloat16,
        device="mcpu",
        routing_method=RoutingMethodType.TopK,
        activation=MoEActivation.SILU,
    )
    mgr = ExpertMapManager(
        max_num_batched_tokens=4,
        top_k=2,
        global_num_experts=10,
        num_redundant_experts=2,
        num_expert_group=None,
        moe_parallel_config=moe_config.moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=True,
    )
    with set_current_vllm_config(VllmConfig()):
        layer = XcpuRoutedExperts(
            layer_name="model.layers.0.mlp.experts",
            params_dtype=torch.bfloat16,
            moe_config=moe_config,
            quant_config=None,
            expert_map_manager=mgr,
        )
    layer.is_fused_checkpoint_transposed = False
    # Fill params with sentinel values so unloaded slots would fail loudly.
    layer.w13_weight.data.fill_(-999.0)
    layer.w2_weight.data.fill_(-999.0)

    # 8 logical experts, 2 redundant replicas -> EP rank 1 owns physical 5..9:
    # primary logical [5, 6, 7] in local slots 0..2 and redundant [0, 1] in slots 3..4.
    gate_up_proj = torch.arange(8 * 16 * 16, dtype=torch.bfloat16).reshape(8, 16, 16)
    down_proj = torch.arange(8 * 16 * 8, dtype=torch.bfloat16).reshape(8, 16, 8)
    loaded = set(
        layer.load_weights([("gate_up_proj", gate_up_proj), ("down_proj", down_proj)])
    )
    assert loaded == {"w13_weight", "w2_weight"}
    expected_logical = [5, 6, 7, 0, 1]
    for local_idx, logical_idx in enumerate(expected_logical):
        torch.testing.assert_close(
            layer.w13_weight.data[local_idx].cpu(),
            gate_up_proj[logical_idx],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer.w2_weight.data[local_idx].cpu(),
            down_proj[logical_idx],
            rtol=0,
            atol=0,
        )
