# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def test_reduce_scatterv() -> None:
    _distributed_run(_reduce_scatterv_worker)
