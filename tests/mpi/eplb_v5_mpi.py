# SPDX-License-Identifier: Apache-2.0
"""Multi-rank MPI integration test for XCPU EPLB on fused EP MoE v5/v6.

Verifies MpiEplbCommunicator (mpi) and TorchDistGlooStagedEplbCommunicator (torch_gloo)
across BF16, FP8 (tight & PaddingExample), and MXFP4 (PaddingExample) including full
forward numerical equivalence before vs after cross-rank weight rebalance.

Run with:
  mpirun --allow-run-as-root --bind-to none -np 2 \
    python tests/mpi/eplb_v5_mpi.py
"""

# MPI must be initialized before importing torch_mcpu / vllm.
# ruff: noqa: E402

from __future__ import annotations

import os
import socket
import time
from types import SimpleNamespace

import mpi4py
import numpy as np

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
if provided != MPI.THREAD_MULTIPLE:
    raise RuntimeError(f"Expected MPI_THREAD_MULTIPLE, got {provided}")

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ["VLLM_CPU_USE_MPI"] = "1"
os.environ["VLLM_XCPU_ENABLE_ENGRAM_PARALLEL"] = "0"
os.environ["VLLM_XCPU_ENABLE_ENGRAM_DISAGG"] = "0"
os.environ["TORCH_XCPU_ENABLE_CHECK"] = "1"

import torch
import torch_mcpu  # noqa: F401
import torch_xcpu
import vllm.model_executor.layers.fused_moe.router.base_router as base_router
from torch_xcpu import ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_communicator import (
    TorchDistGlooStagedEplbCommunicator,
    create_eplb_communicator,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState, EplbState
from vllm.distributed.eplb.rebalance_execute import (
    get_ep_ranks_with_experts_batch,
    move_from_buffer,
    rearrange_expert_weights_inplace,
    transfer_layer,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_ep_group,
    get_eplb_group,
    init_distributed_environment,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.utils import replace_parameter

import vllm_xcpu_plugin
from vllm_xcpu_plugin.distributed.mpi_eplb_communicator import MpiEplbCommunicator
from vllm_xcpu_plugin.distributed.mpi_world import (
    ClusterType,
    initialize_mpi_world,
)
from vllm_xcpu_plugin.layers.fused_moe.ep_experts import XcpuEPExperts
from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_logical_mapping(
    physical_to_logical: torch.Tensor, num_logical: int
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = [[] for _ in range(num_logical)]
    for phys_id, log_id in enumerate(physical_to_logical.tolist()):
        slots[log_id].append(phys_id)
    max_slots = max(len(s) for s in slots)
    l2p = torch.full((num_logical, max_slots), -1, dtype=torch.int64, device="cpu")
    counts = torch.zeros(num_logical, dtype=torch.int64, device="cpu")
    for log_id, phys_list in enumerate(slots):
        counts[log_id] = len(phys_list)
        for idx, phys_id in enumerate(phys_list):
            l2p[log_id, idx] = phys_id
    return l2p.to("mcpu"), counts.to("mcpu")


def _expected_source_location(
    old_placement: list[int],
    new_placement: list[int],
    *,
    ep_rank: int,
    local_slot: int,
    local_experts: int,
) -> tuple[int, int]:
    """Return the (src_rank, src_slot) whose bytes fill (ep_rank, local_slot)."""
    old_local = old_placement[ep_rank * local_experts : (ep_rank + 1) * local_experts]
    new_local = new_placement[ep_rank * local_experts : (ep_rank + 1) * local_experts]
    log_id = new_local[local_slot]
    if old_local[local_slot] == log_id:
        return ep_rank, local_slot
    if log_id in old_local:
        return ep_rank, old_local.index(log_id)
    send_map, recv_map = get_ep_ranks_with_experts_batch(
        np.asarray([log_id], dtype=np.int64),
        local_experts,
        np.asarray(old_placement, dtype=np.int64),
        np.asarray(new_placement, dtype=np.int64),
    )
    ranks_to_send = send_map[log_id]
    ranks_to_recv = recv_map[log_id]
    num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
    recver_pos = ranks_to_recv.index(ep_rank)
    remainder_start = len(ranks_to_send) * num_dst_per_sender
    if recver_pos < remainder_start:
        src_rank = ranks_to_send[recver_pos // num_dst_per_sender]
    else:
        src_rank = ranks_to_send[recver_pos - remainder_start]
    src_local = old_placement[
        src_rank * local_experts : (src_rank + 1) * local_experts
    ]
    return int(src_rank), src_local.index(log_id)


def _make_routed_layer(
    *,
    num_physical: int,
    num_logical: int,
    ep_size: int,
    ep_rank: int,
    hidden: int,
    intermediate: int,
    topk: int,
    max_tokens: int,
    version: int,
    fused_moe,
    quant_config=None,
    scale_names: tuple[str, str] | None = None,
) -> XcpuRoutedExperts:
    local_experts = num_physical // ep_size
    moe_cfg = FusedMoEConfig(
        num_experts=num_physical,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=local_experts,
        num_logical_experts=num_logical,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            dp_size=ep_size,
            dp_rank=ep_rank,
            pcp_size=1,
            pcp_rank=0,
            ep_size=ep_size,
            ep_rank=ep_rank,
            sp_size=1,
            use_ep=True,
            enable_eplb=True,
            all2all_backend=f"mpi_alltoallv_v{version}",
        ),
        in_dtype=torch.bfloat16,
        device="mcpu",
        routing_method=RoutingMethodType.TopK,
        activation=MoEActivation.SILU,
    )
    mgr = ExpertMapManager(
        max_num_batched_tokens=max_tokens,
        top_k=topk,
        global_num_experts=num_physical,
        num_redundant_experts=num_physical - num_logical,
        num_expert_group=None,
        moe_parallel_config=moe_cfg.moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=True,
    )
    layer = XcpuRoutedExperts(
        layer_name="test_moe",
        params_dtype=torch.bfloat16,
        moe_config=moe_cfg,
        quant_config=quant_config,
        expert_map_manager=mgr,
    )
    gemm1 = fused_moe.params.gemm1.params
    gemm2 = fused_moe.params.gemm2.params
    replace_parameter(layer, "w13_weight", gemm1.packed_weight)
    replace_parameter(layer, "w2_weight", gemm2.packed_weight)
    if scale_names is not None:
        replace_parameter(layer, scale_names[0], gemm1.packed_weight_scale)
        replace_parameter(layer, scale_names[1], gemm2.packed_weight_scale)
    layer._xcpu_fused_moe = fused_moe
    return layer


def _init_weights_and_scales(
    fmt: str,
    backend: ops.MoeGroupedGemmBackend,
    *,
    num_logical: int,
    hidden: int,
    intermediate: int,
    logical_placement: list[int],
):
    gen = torch.Generator(device="cpu").manual_seed(2026)
    indices = torch.tensor(logical_placement, dtype=torch.int64, device="cpu")
    scale_names: tuple[str, str] | None = None
    block_size: tuple[int, int] | None = None

    if fmt == "bf16":
        w13_all = (
            torch.randn(num_logical, 2 * intermediate, hidden, generator=gen) * 0.1
        ).to(torch.bfloat16)
        w2_all = (
            torch.randn(num_logical, hidden, intermediate, generator=gen) * 0.1
        ).to(torch.bfloat16)
        w13_s, w2_s = None, None
    elif fmt == "fp8":
        block_size = (128, 128)
        w13_all = (
            torch.randn(num_logical, 2 * intermediate, hidden, generator=gen) * 0.2
        ).to(torch.float8_e4m3fn)
        w2_all = (
            torch.randn(num_logical, hidden, intermediate, generator=gen) * 0.2
        ).to(torch.float8_e4m3fn)
        w13_s = (
            torch.rand(
                num_logical,
                (2 * intermediate + 127) // 128,
                (hidden + 127) // 128,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.5
            + 0.25
        )
        w2_s = (
            torch.rand(
                num_logical,
                (hidden + 127) // 128,
                (intermediate + 127) // 128,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.5
            + 0.25
        )
        scale_names = ("w13_weight_scale_inv", "w2_weight_scale_inv")
    elif fmt == "mxfp4":
        block_size = (1, 32)
        # Balanced signed E2M1 nibbles with moderate E8M0 exponents keep outputs ~O(1).
        w13_all = torch.randint(
            0,
            256,
            (num_logical, 2 * intermediate, hidden // 2),
            generator=gen,
            dtype=torch.uint8,
        )
        w2_all = torch.randint(
            0,
            256,
            (num_logical, hidden, intermediate // 2),
            generator=gen,
            dtype=torch.uint8,
        )
        w13_s = torch.randint(
            123,
            126,
            (num_logical, 2 * intermediate, hidden // 32),
            generator=gen,
            dtype=torch.uint8,
        )
        w2_s = torch.randint(
            123,
            126,
            (num_logical, hidden, intermediate // 32),
            generator=gen,
            dtype=torch.uint8,
        )
        scale_names = ("w13_weight_scale", "w2_weight_scale")
    else:
        raise ValueError(fmt)

    w13_local = w13_all[indices].contiguous().to("mcpu")
    w2_local = w2_all[indices].contiguous().to("mcpu")
    w13_s_local = None if w13_s is None else w13_s[indices].contiguous().to("mcpu")
    w2_s_local = None if w2_s is None else w2_s[indices].contiguous().to("mcpu")
    fused_moe = ops.initialize_fused_moe(
        w13_local,
        w2_local,
        w1_scale=w13_s_local,
        w2_scale=w2_s_local,
        scale_block_size=block_size,
        backend=backend,
        allocate_scratch=False,
    )
    return fused_moe, scale_names


def _run_rebalance(
    *,
    old_indices: torch.Tensor,
    new_indices: torch.Tensor,
    expert_weights: list[list[torch.Tensor]],
    expert_buffer: list[torch.Tensor],
    ep_group,
    comm,
    async_stream: torch.Stream | None,
) -> None:
    if async_stream is None:
        rearrange_expert_weights_inplace(
            old_indices,
            new_indices,
            expert_weights,
            expert_buffer,
            ep_group.device_group,
            comm,
            is_profile=False,
        )
        return
    # Mirror async_worker.py + eplb_state._move_to_workspace handoff.
    comm.set_stream(async_stream)
    try:
        transfer_metadata = transfer_layer(
            old_layer_indices=old_indices[0],
            new_layer_indices=new_indices[0],
            expert_weights=expert_weights[0],
            expert_weights_buffer=expert_buffer,
            ep_group=ep_group.device_group,
            communicator=comm,
            is_profile=False,
            stream=async_stream,
        )
        async_stream.synchronize()
    finally:
        comm.set_stream(None)
    move_from_buffer(
        expert_weights=expert_weights[0],
        expert_weights_buffers=expert_buffer,
        transfer_metadata=transfer_metadata,
        new_indices=new_indices[0].cpu().numpy(),
        ep_rank=ep_group.rank_in_group,
    )


def _verify_case(
    *,
    rank: int,
    world_size: int,
    ep_group,
    comm_name: str,
    comm,
    version: int,
    fmt: str,
    backend: ops.MoeGroupedGemmBackend,
    async_stream: torch.Stream | None = None,
) -> None:
    num_logical = 6
    num_physical = 8
    local_experts = num_physical // world_size
    hidden = 256
    # Use K=256 (FP8) / K=128 (MXFP4) so PaddingExample allocates extra padding bytes.
    if backend == ops.MoeGroupedGemmBackend.INTEL_AMX:
        intermediate = 128
    elif backend == ops.MoeGroupedGemmBackend.PADDING_EXAMPLE:
        if fmt == "mxfp4":
            hidden = 128
            intermediate = 128
        else:
            intermediate = 256
    else:
        intermediate = 64
    topk = 3
    max_tokens = 4

    old_placement = [0, 1, 2, 3, 4, 5, 0, 1]
    new_placement = [5, 3, 2, 0, 1, 4, 2, 5]

    old_indices = torch.tensor([old_placement], dtype=torch.int64, device="mcpu")
    new_indices = torch.tensor([new_placement], dtype=torch.int64, device="mcpu")

    local_slice = old_placement[rank * local_experts : (rank + 1) * local_experts]
    fused_moe, scale_names = _init_weights_and_scales(
        fmt,
        backend,
        num_logical=num_logical,
        hidden=hidden,
        intermediate=intermediate,
        logical_placement=local_slice,
    )
    layer = _make_routed_layer(
        num_physical=num_physical,
        num_logical=num_logical,
        ep_size=world_size,
        ep_rank=rank,
        hidden=hidden,
        intermediate=intermediate,
        topk=topk,
        max_tokens=max_tokens,
        version=version,
        fused_moe=fused_moe,
        scale_names=scale_names,
    )

    expert_weights = [list(layer.get_expert_weights())]
    for t in expert_weights[0]:
        assert (
            t.dtype == torch.uint8 and t.shape[0] == local_experts and t.is_contiguous()
        )
    expert_buffer = [torch.empty_like(t) for t in expert_weights[0]]

    # Snapshot exact pre-rebalance bytes of every physical replica across ranks so
    # even uninitialized trailing bytes inside PaddingExample storage match bit-for-bit
    # against the exact physical replica chosen by rebalance_execute.
    local_snapshot = [t.cpu().clone() for t in expert_weights[0]]
    all_snapshots = MPI.COMM_WORLD.allgather(local_snapshot)

    ep_dev_comm = ep_group.device_communicator
    comm_metadata = torch.tensor([world_size, rank], dtype=torch.int64, device="cpu")
    expert_map = torch.full((num_physical,), -1, dtype=torch.int32, device="mcpu")
    expert_map[rank * local_experts : (rank + 1) * local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device="mcpu"
    )
    ep_experts = XcpuEPExperts(
        fused_moe,
        comm_metadata,
        ep_dev_comm.comm_ptr_wrapper,
        expert_map,
        version=version,
        ep_size=world_size,
        ep_rank=rank,
        max_num_tokens=max_tokens,
        topk=topk,
    )
    ep_experts.initialize_windows()

    gen = torch.Generator(device="cpu").manual_seed(100 + rank)
    hidden_states = (torch.randn(max_tokens, hidden, generator=gen) * 0.25).to(
        torch.bfloat16
    ).to("mcpu")
    topk_weights = torch.full(
        (max_tokens, topk), 1.0 / topk, dtype=torch.float32, device="mcpu"
    )
    topk_ids_logical = torch.tensor(
        [[0, 5, 1], [1, 4, 2], [2, 3, 0], [5, 0, 4]]
        if rank == 0
        else [[3, 2, 5], [4, 1, 0], [5, 2, 3], [0, 3, 1]],
        dtype=torch.int32,
        device="mcpu",
    )

    old_l2p, old_cnt = _build_logical_mapping(old_indices[0].cpu(), num_logical)
    load_view = torch.zeros(num_physical, dtype=torch.int32, device="mcpu")
    rec_t = torch.tensor(True, dtype=torch.bool, device="mcpu")
    phys_before = base_router.eplb_map_to_physical_and_record(
        topk_ids=topk_ids_logical,
        expert_load_view=load_view,
        logical_to_physical_map=old_l2p,
        logical_replica_count=old_cnt,
        record_enabled=rec_t,
    )
    out_before = ep_experts.forward(hidden_states, topk_weights, phys_before)
    torch.accelerator.synchronize()
    ref_out = out_before.cpu().clone()

    if comm.needs_profile_buffer_reservation:
        rearrange_expert_weights_inplace(
            old_indices,
            old_indices,
            expert_weights,
            expert_buffer,
            ep_group.device_group,
            comm,
            is_profile=True,
        )

    _run_rebalance(
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=expert_weights,
        expert_buffer=expert_buffer,
        ep_group=ep_group,
        comm=comm,
        async_stream=async_stream,
    )
    torch.accelerator.synchronize()

    # Check bit-for-bit byte equality against the exact source physical replica.
    for t_idx, actual_t in enumerate(expert_weights[0]):
        actual_cpu = actual_t.cpu()
        for local_slot in range(local_experts):
            src_rank, src_slot = _expected_source_location(
                old_placement,
                new_placement,
                ep_rank=rank,
                local_slot=local_slot,
                local_experts=local_experts,
            )
            assert torch.equal(
                actual_cpu[local_slot], all_snapshots[src_rank][t_idx][src_slot]
            ), (
                f"Weight mismatch at tensor {t_idx} slot {local_slot} "
                f"(src=({src_rank},{src_slot})) on rank {rank} for {comm_name} "
                f"v{version} {fmt}"
            )

    new_l2p, new_cnt = _build_logical_mapping(new_indices[0].cpu(), num_logical)
    phys_after = base_router.eplb_map_to_physical_and_record(
        topk_ids=topk_ids_logical,
        expert_load_view=load_view,
        logical_to_physical_map=new_l2p,
        logical_replica_count=new_cnt,
        record_enabled=rec_t,
    )
    out_after = ep_experts.forward(hidden_states, topk_weights, phys_after)
    torch.accelerator.synchronize()

    rtol, atol = (1e-1, 3e-2) if fmt == "mxfp4" else (2e-2, 5e-3)
    torch.testing.assert_close(out_after.cpu(), ref_out, rtol=rtol, atol=atol)

    # Round-trip back to old_indices to verify bit-for-bit exact output restoration.
    _run_rebalance(
        old_indices=new_indices,
        new_indices=old_indices,
        expert_weights=expert_weights,
        expert_buffer=expert_buffer,
        ep_group=ep_group,
        comm=comm,
        async_stream=async_stream,
    )
    torch.accelerator.synchronize()
    out_roundtrip = ep_experts.forward(hidden_states, topk_weights, phys_before)
    torch.accelerator.synchronize()
    assert torch.equal(out_roundtrip.cpu(), ref_out), (
        f"Round-trip output mismatch on rank {rank} for {comm_name} v{version} {fmt}: "
        f"max_diff={(out_roundtrip.cpu() - ref_out).abs().max().item()}"
    )
    if rank == 0:
        print(
            f"[PASS] comm={comm_name} v{version} fmt={fmt} backend={backend.name} "
            f"async={async_stream is not None}"
        )


def main() -> None:
    world_comm = MPI.COMM_WORLD
    rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()
    if world_size < 2 or world_size % 2 != 0:
        raise RuntimeError(f"Expected even world_size >= 2, got {world_size}")

    port = _find_free_port() if rank == 0 else None
    port = world_comm.bcast(port, root=0)
    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })

    vllm_xcpu_plugin.xcpu_platform_plugin()
    vllm_xcpu_plugin.register_ops()
    initialize_mpi_world(ClusterType.ATTN)
    torch_xcpu.initialize_runtime()

    parallel_cfg = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True,
        all2all_backend="mpi_alltoallv_v5",
        enable_eplb=True,
        eplb_config=EPLBConfig(num_redundant_experts=2, communicator="mpi"),
    )
    vllm_config = VllmConfig(parallel_config=parallel_cfg)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="cpu:gloo,mcpu:mcpu",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="cpu:gloo,mcpu:mcpu",
        )

        eplb_group = get_eplb_group()
        ep_group = get_ep_group()
        mpi_comm = create_eplb_communicator(eplb_group, "mpi", [], [])
        gloo_comm = create_eplb_communicator(eplb_group, "torch_gloo", [], [])
        assert isinstance(mpi_comm, MpiEplbCommunicator)
        assert isinstance(gloo_comm, TorchDistGlooStagedEplbCommunicator)
        assert mpi_comm.needs_profile_buffer_reservation is False
        assert gloo_comm.needs_profile_buffer_reservation is True

        fp8_backend = (
            ops.MoeGroupedGemmBackend.INTEL_AMX
            if ops.moe_grouped_gemm_backend_available(
                ops.MoeWeightFormat.FP8_E4M3, ops.MoeGroupedGemmBackend.INTEL_AMX
            )
            else ops.MoeGroupedGemmBackend.PORTABLE_LUT
        )
        cases = [
            (5, "bf16", ops.MoeGroupedGemmBackend.PORTABLE, None),
            (6, "bf16", ops.MoeGroupedGemmBackend.PORTABLE, None),
            (5, "fp8", fp8_backend, None),
            (6, "fp8", ops.MoeGroupedGemmBackend.PADDING_EXAMPLE, None),
            (5, "mxfp4", ops.MoeGroupedGemmBackend.PADDING_EXAMPLE, None),
            (
                5,
                "fp8",
                ops.MoeGroupedGemmBackend.PADDING_EXAMPLE,
                torch.Stream(device="mcpu"),
            ),
        ]
        for comm_name, comm in [("mpi", mpi_comm), ("torch_gloo", gloo_comm)]:
            for version, fmt, backend, stream in cases:
                _verify_case(
                    rank=rank,
                    world_size=world_size,
                    ep_group=ep_group,
                    comm_name=comm_name,
                    comm=comm,
                    version=version,
                    fmt=fmt,
                    backend=backend,
                    async_stream=stream,
                )

        for use_async in (False, True):
            _verify_eplb_state_lifecycle(
                rank=rank,
                world_size=world_size,
                ep_group=ep_group,
                use_async=use_async,
            )

    world_comm.Barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    MPI.Finalize()


class _LifecycleMoEModel(torch.nn.Module):
    def __init__(
        self,
        layer: XcpuRoutedExperts,
        *,
        num_logical: int,
        num_physical: int,
        local_experts: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_routed_experts = num_logical
        self.num_redundant_experts = num_physical - num_logical
        self.num_physical_experts = num_physical
        self.num_local_physical_experts = local_experts
        self.num_logical_experts = num_logical
        self.num_expert_groups = 1
        self.num_moe_layers = 1
        self.moe_layers = [layer]
        self.expert_weights = [list(layer.get_expert_weights())]

    def set_eplb_state(
        self,
        expert_load_pass: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.layer.eplb_state.set_layer_state(
            0,
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )


def _verify_eplb_state_lifecycle(
    *,
    rank: int,
    world_size: int,
    ep_group,
    use_async: bool,
) -> None:
    num_logical = 6
    num_redundant = 2
    num_physical = num_logical + num_redundant
    local_experts = num_physical // world_size
    hidden = 256
    intermediate = 64
    topk = 3
    max_tokens = 4
    version = 5

    initial_placement = list(
        EplbState.build_initial_global_physical_to_logical_map(
            num_logical, num_redundant
        )
    )
    local_slice = initial_placement[rank * local_experts : (rank + 1) * local_experts]
    fused_moe, scale_names = _init_weights_and_scales(
        "bf16",
        ops.MoeGroupedGemmBackend.PORTABLE,
        num_logical=num_logical,
        hidden=hidden,
        intermediate=intermediate,
        logical_placement=local_slice,
    )
    layer = _make_routed_layer(
        num_physical=num_physical,
        num_logical=num_logical,
        ep_size=world_size,
        ep_rank=rank,
        hidden=hidden,
        intermediate=intermediate,
        topk=topk,
        max_tokens=max_tokens,
        version=version,
        fused_moe=fused_moe,
        scale_names=scale_names,
    )
    layer.eplb_state = EplbLayerState()

    ep_dev_comm = ep_group.device_communicator
    comm_metadata = torch.tensor([world_size, rank], dtype=torch.int64, device="cpu")
    expert_map = torch.full((num_physical,), -1, dtype=torch.int32, device="mcpu")
    expert_map[rank * local_experts : (rank + 1) * local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device="mcpu"
    )
    ep_experts = XcpuEPExperts(
        fused_moe,
        comm_metadata,
        ep_dev_comm.comm_ptr_wrapper,
        expert_map,
        version=version,
        ep_size=world_size,
        ep_rank=rank,
        max_num_tokens=max_tokens,
        topk=topk,
    )
    ep_experts.initialize_windows()

    model = _LifecycleMoEModel(
        layer,
        num_logical=num_logical,
        num_physical=num_physical,
        local_experts=local_experts,
    )
    state_parallel_cfg = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True,
        all2all_backend="mpi_alltoallv_v5",
        enable_eplb=True,
        eplb_config=EPLBConfig(
            num_redundant_experts=num_redundant,
            window_size=2,
            step_interval=8,
            log_balancedness_interval=2,
            use_async=use_async,
            communicator="mpi",
        ),
    )
    mode_tag = "async" if use_async else "sync"
    model_cfg = SimpleNamespace(
        model=f"lifecycle_moe_{mode_tag}",
        compute_hash=lambda tag=mode_tag: f"lifecycle_hash_{tag}",
    )
    eplb_state = EplbState(state_parallel_cfg, torch.device("mcpu:0"))
    eplb_state.add_model(model, model_cfg)
    model_state = eplb_state.model_states[model_cfg.compute_hash()]
    assert isinstance(model_state.communicator, MpiEplbCommunicator)

    gen = torch.Generator(device="cpu").manual_seed(300 + rank)
    hidden_states = (
        (torch.randn(max_tokens, hidden, generator=gen) * 0.25)
        .to(torch.bfloat16)
        .to("mcpu")
    )
    topk_weights = torch.full(
        (max_tokens, topk), 1.0 / topk, dtype=torch.float32, device="mcpu"
    )
    eval_topk_ids = torch.tensor(
        [[0, 5, 1], [1, 4, 2], [2, 3, 0], [5, 0, 4]]
        if rank == 0
        else [[3, 2, 5], [4, 1, 0], [5, 2, 3], [0, 3, 1]],
        dtype=torch.int32,
        device="mcpu",
    )
    # Heavily skew load toward logical experts 4 and 5 (initially 1 replica each).
    skewed_topk_ids = torch.tensor(
        [[4, 5, 2], [4, 5, 3], [4, 5, 2], [4, 5, 3]],
        dtype=torch.int32,
        device="mcpu",
    )

    def _forward_with_eplb(topk_ids: torch.Tensor) -> torch.Tensor:
        ls = layer.eplb_state
        assert ls.expert_load_view is not None
        assert ls.logical_to_physical_map is not None
        assert ls.logical_replica_count is not None
        assert ls.should_record_tensor is not None
        assert ls.num_unpadded_tokens_tensors is not None
        eplb_state.prepare_forward(model_cfg, num_unpadded_tokens=max_tokens)
        phys_ids = base_router.eplb_map_to_physical_and_record(
            topk_ids=topk_ids,
            expert_load_view=ls.expert_load_view,
            logical_to_physical_map=ls.logical_to_physical_map,
            logical_replica_count=ls.logical_replica_count,
            record_enabled=ls.should_record_tensor,
            num_unpadded_tokens=ls.num_unpadded_tokens_tensors[0],
        )
        return ep_experts.forward(hidden_states, topk_weights, phys_ids)

    ref_eval_out = _forward_with_eplb(eval_topk_ids)
    torch.accelerator.synchronize()
    ref_eval_cpu = ref_eval_out.cpu().clone()
    model_state.expert_load_pass.zero_()

    if not use_async:
        # Exercise profile rearrange + torch.Event.elapsed_time path.
        eplb_state.step(is_profile=True)
    else:
        eplb_state.start_async_loop()

    initial_map_cpu = model_state.physical_to_logical_map.cpu().clone()

    # Initial expert_rearrangement_step is 6 (step_interval=8, window_size=2), so
    # 2 skewed steps fill the sliding window and trigger rearrange() on step 8.
    for _ in range(2):
        _ = _forward_with_eplb(skewed_topk_ids)
        torch.accelerator.synchronize()
        eplb_state.step(log_stats=True)

    if use_async:
        for _ in range(500):
            if not model_state.rebalanced:
                break
            time.sleep(0.01)
            eplb_state.expert_rearrangement_step = 0
            eplb_state.step(is_dummy=True)
        assert not model_state.rebalanced, "Async EPLB did not finish within timeout"

    torch.accelerator.synchronize()
    updated_map_cpu = model_state.physical_to_logical_map.cpu()
    assert not torch.equal(updated_map_cpu, initial_map_cpu)
    assert int((updated_map_cpu[0] == 4).sum().item()) == 2
    assert int((updated_map_cpu[0] == 5).sum().item()) == 2

    out_after = _forward_with_eplb(eval_topk_ids)
    torch.accelerator.synchronize()
    torch.testing.assert_close(out_after.cpu(), ref_eval_cpu, rtol=2e-2, atol=5e-3)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print(f"[PASS] EplbState lifecycle mode={mode_tag}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
