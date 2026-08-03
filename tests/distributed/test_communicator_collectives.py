# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable

import multiprocess as mp
import torch
import torch.distributed as dist
from vllm.utils.network_utils import get_open_port

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import (
    CpuMPICommunicator,
)
from vllm_xcpu_plugin.distributed.xcpu_communicator import CpuCommunicator

WORLD_SIZE = 2
COMMUNICATOR_CLASSES = (CpuCommunicator, CpuMPICommunicator)


def _all_gather(input_: torch.Tensor, dim: int) -> torch.Tensor:
    outputs = [torch.empty_like(input_) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, input_)
    return torch.cat(outputs, dim=dim)


def _make_communicator(
    communicator_cls: type[CpuCommunicator] | type[CpuMPICommunicator],
) -> CpuCommunicator | CpuMPICommunicator:
    communicator = object.__new__(communicator_cls)
    communicator.rank_in_group = dist.get_rank()
    communicator.world_size = dist.get_world_size()
    communicator.device_group = dist.group.WORLD

    if isinstance(communicator, CpuCommunicator):
        communicator.dist_module = dist
    else:
        # CpuMPICommunicator.all_gather is backed by torch_xcpu/MPI. These tests
        # exercise all_gatherv above that equal-size collective boundary.
        communicator.all_gather = _all_gather

    return communicator


def _all_gatherv_worker(rank: int, port: int) -> None:
    _init_process_group(rank, port)
    try:
        sizes = [3, 2]
        inputs = [
            torch.arange(2 * size, dtype=torch.float32).reshape(2, size)
            + process_rank * 100
            for process_rank, size in enumerate(sizes)
        ]
        expected = torch.cat(inputs, dim=1)

        for communicator_cls in COMMUNICATOR_CLASSES:
            communicator = _make_communicator(communicator_cls)
            local_input = inputs[rank]

            actual = communicator.all_gatherv(local_input, dim=1, sizes=sizes)
            torch.testing.assert_close(actual, expected)

            actual_list = communicator.all_gatherv(
                [local_input, local_input + 1000],
                dim=1,
                sizes=sizes,
            )
            assert isinstance(actual_list, list)
            torch.testing.assert_close(actual_list[0], expected)
            torch.testing.assert_close(actual_list[1], expected + 1000)
    finally:
        dist.destroy_process_group()


def _reduce_scatter_worker(rank: int, port: int) -> None:
    _init_process_group(rank, port)
    try:
        inputs = [
            torch.arange(8, dtype=torch.float32).reshape(2, 4) + process_rank * 100
            for process_rank in range(WORLD_SIZE)
        ]
        reduced = sum(inputs)
        expected = reduced.chunk(WORLD_SIZE, dim=1)[rank]

        for communicator_cls in COMMUNICATOR_CLASSES:
            communicator = _make_communicator(communicator_cls)
            actual = communicator.reduce_scatter(inputs[rank], dim=1)
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def _mcpu_sequence_parallel_collectives_worker(rank: int, port: int) -> None:
    dist.init_process_group(
        "cpu:gloo,mcpu:mcpu",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        import torch_mcpu  # noqa: F401

        inputs = [
            torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
            + process_rank * 100
            for process_rank in range(WORLD_SIZE)
        ]
        expected_full = sum(inputs)
        expected_chunk = expected_full.chunk(WORLD_SIZE, dim=0)[rank]

        communicator = _make_communicator(CpuCommunicator)
        actual_chunk = communicator.reduce_scatter(inputs[rank].to("mcpu"), dim=0)
        torch.testing.assert_close(actual_chunk.cpu(), expected_chunk)

        actual_full = communicator.all_gather(actual_chunk, dim=0)
        expected_gathered = torch.cat(
            expected_full.chunk(WORLD_SIZE, dim=0), dim=0
        )
        torch.testing.assert_close(actual_full.cpu(), expected_gathered)
    finally:
        dist.destroy_process_group()


def _mcpu_replicated_shared_expert_worker(rank: int, port: int) -> None:
    # Sequence-parallel Qwen3.5/3.6 uses a fully replicated shared-expert
    # MLP on each TP rank. Exercise the same disable_tp=True construction and
    # checkpoint loaders used by Qwen2MoeMLP/Qwen3NextMLP.
    os.environ["VLLM_CPU_USE_MPI"] = "0"

    import torch.nn.functional as F
    import torch_mcpu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
        try:
            torch.set_default_device("mcpu")
            torch.set_default_dtype(torch.bfloat16)

            hidden_size = 16
            intermediate_size = 8
            expert_gate = ReplicatedLinear(
                hidden_size,
                1,
                bias=False,
                prefix="shared_expert_gate",
            )
            layer = Qwen2MoeMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                reduce_results=False,
                expert_gate=expert_gate,
                is_sequence_parallel=True,
                prefix="shared_expert",
            )

            gate_weight = (
                torch.arange(hidden_size, dtype=torch.float32, device="cpu")
                .reshape(1, hidden_size)
                .mul_(0.002)
                .to(torch.bfloat16)
            )
            gate_proj_weight = (
                torch.arange(
                    intermediate_size * hidden_size,
                    dtype=torch.float32,
                    device="cpu",
                )
                .reshape(intermediate_size, hidden_size)
                .sub_(64)
                .mul_(0.001)
                .to(torch.bfloat16)
            )
            up_proj_weight = gate_proj_weight.flip(0).contiguous()
            down_proj_weight = (
                torch.arange(
                    hidden_size * intermediate_size,
                    dtype=torch.float32,
                    device="cpu",
                )
                .reshape(hidden_size, intermediate_size)
                .sub_(48)
                .mul_(0.0015)
                .to(torch.bfloat16)
            )

            expert_gate.weight_loader(expert_gate.weight, gate_weight)
            layer.gate_up_proj.weight_loader(
                layer.gate_up_proj.weight, gate_proj_weight, 0
            )
            layer.gate_up_proj.weight_loader(
                layer.gate_up_proj.weight, up_proj_weight, 1
            )
            layer.down_proj.weight_loader(
                layer.down_proj.weight, down_proj_weight
            )

            assert layer.gate_up_proj.tp_size == 1
            assert layer.gate_up_proj.tp_rank == 0
            assert layer.down_proj.tp_size == 1
            assert layer.down_proj.tp_rank == 0
            assert tuple(layer.gate_up_proj.weight.shape) == (
                2 * intermediate_size,
                hidden_size,
            )
            assert tuple(layer.down_proj.weight.shape) == (
                hidden_size,
                intermediate_size,
            )

            # Each rank receives a different sequence chunk, as it does in SP.
            x_cpu = (
                torch.arange(3 * hidden_size, dtype=torch.float32, device="cpu")
                .reshape(3, hidden_size)
                .add_(rank * 17)
                .mul_(0.01)
                .to(torch.bfloat16)
            )
            actual = layer(x_cpu.to("mcpu"))
            torch.accelerator.synchronize()

            gate_up = F.linear(
                x_cpu,
                torch.cat([gate_proj_weight, up_proj_weight], dim=0),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expected = F.linear(F.silu(gate) * up, down_proj_weight)
            expected *= torch.sigmoid(F.linear(x_cpu, gate_weight))
            torch.testing.assert_close(
                actual.cpu(), expected, atol=2e-3, rtol=2e-2
            )

            # Replicated weights must be byte-identical across TP ranks.
            for parameter in (
                expert_gate.weight,
                layer.gate_up_proj.weight,
                layer.down_proj.weight,
            ):
                local = parameter.detach().cpu()
                gathered = [torch.empty_like(local) for _ in range(WORLD_SIZE)]
                dist.all_gather(gathered, local)
                for replica in gathered[1:]:
                    torch.testing.assert_close(replica, gathered[0], rtol=0, atol=0)
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)
            destroy_model_parallel()
            destroy_distributed_environment()


def _reduce_scatterv_worker(rank: int, port: int) -> None:
    _init_process_group(rank, port)
    try:
        sizes = [1, 2]
        inputs = [
            torch.arange(6, dtype=torch.float32).reshape(2, 3) + process_rank * 100
            for process_rank in range(WORLD_SIZE)
        ]
        reduced = sum(inputs)
        offset = sum(sizes[:rank])
        expected = reduced.narrow(1, offset, sizes[rank])

        for communicator_cls in COMMUNICATOR_CLASSES:
            communicator = _make_communicator(communicator_cls)
            actual = communicator.reduce_scatterv(
                inputs[rank],
                dim=1,
                sizes=sizes,
            )
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def _init_process_group(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )


def _distributed_run(worker: Callable[[int, int], None]) -> None:
    context = mp.get_context("spawn")
    port = get_open_port()
    processes = [
        context.Process(target=worker, args=(rank, port)) for rank in range(WORLD_SIZE)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


def test_all_gatherv() -> None:
    _distributed_run(_all_gatherv_worker)


def test_reduce_scatter() -> None:
    _distributed_run(_reduce_scatter_worker)


def test_mcpu_sequence_parallel_collectives() -> None:
    _distributed_run(_mcpu_sequence_parallel_collectives_worker)


def test_mcpu_replicated_shared_expert() -> None:
    _distributed_run(_mcpu_replicated_shared_expert_worker)


def test_reduce_scatterv() -> None:
    _distributed_run(_reduce_scatterv_worker)
