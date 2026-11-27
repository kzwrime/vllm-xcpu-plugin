# SPDX-License-Identifier: Apache-2.0
"""Multi-rank MPI integration tests for CpuMPICommunicator PP P2P ops.

Run with:
  mpirun --allow-run-as-root -np 4 pytest tests/distributed/test_mpi_pp_send_recv.py
"""

# ruff: noqa: E402
from __future__ import annotations

import os
import socket

import pytest

_MPI_WORLD_SIZE = int(
    os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))
)
if _MPI_WORLD_SIZE < 4:
    pytest.skip(
        "requires at least 4 MPI ranks (run under mpirun -np 4)",
        allow_module_level=True,
    )

import atexit

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    atexit.register(
        lambda: (
            MPI.Finalize() if MPI.Is_initialized() and not MPI.Is_finalized() else None
        )
    )

_WORLD = MPI.COMM_WORLD

os.environ.setdefault("VLLM_CPU_USE_MPI", "1")
os.environ.setdefault("VLLM_XCPU_ENABLE_ENGRAM_PARALLEL", "0")
os.environ.setdefault("VLLM_XCPU_ENABLE_ENGRAM_DISAGG", "0")

import torch
import torch_mcpu  # noqa: F401
import vllm  # noqa: F401
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator
from vllm_xcpu_plugin.distributed.mpi_world import ClusterType, initialize_mpi_world


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_cpu_mpi_communicator_pp_send_recv_and_broadcast() -> None:
    rank = _WORLD.Get_rank()
    world_size = 4
    port = _WORLD.bcast(_find_free_port() if rank == 0 else None, root=0)
    assert port is not None

    initialize_mpi_world(ClusterType.ATTN)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
    )
    vllm_config = VllmConfig(parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="cpu:gloo,mcpu:mcpu",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
        try:
            pp_group = get_pp_group()
            tp_group = get_tp_group()
            pp_comm = pp_group.device_communicator
            assert isinstance(pp_comm, CpuMPICommunicator)
            assert pp_comm.supports_custom_tensor_dict is True
            assert pp_comm.supports_custom_pp_broadcast is True
            assert pp_group.world_size == 2
            assert tp_group.world_size == 2

            # 1. Direct send / recv on mcpu
            msg_shape = (6, 8)
            if pp_group.is_first_rank:
                t_send = torch.full(
                    msg_shape,
                    float(tp_group.rank_in_group + 1),
                    dtype=torch.bfloat16,
                    device="mcpu",
                )
                pp_group.send(t_send)
            else:
                t_recv = pp_group.recv(torch.Size(msg_shape), torch.bfloat16)
                torch.mcpu.synchronize()
                torch.testing.assert_close(
                    t_recv.cpu(),
                    torch.full(
                        msg_shape,
                        float(tp_group.rank_in_group + 1),
                        dtype=torch.bfloat16,
                    ),
                )

            # 2. send_tensor_dict / recv_tensor_dict with and without all_gather_group
            base_hidden = (
                torch
                .arange(32, dtype=torch.float32)
                .reshape(4, 8)
                .to(torch.bfloat16)
                .to("mcpu")
            )
            base_residual = (
                torch.full((4, 8), 7.5, dtype=torch.bfloat16, device="mcpu")
                + tp_group.rank_in_group
            )
            empty_tensor = torch.empty((0,), dtype=torch.int64, device="mcpu")
            payload = {
                "hidden_states": base_hidden,
                "residual": base_residual,
                "empty_field": empty_tensor,
                "step_tag": 42,
            }
            all_gather_map = {"hidden_states": True, "residual": False}

            if pp_group.is_first_rank:
                handles = pp_group.isend_tensor_dict(
                    payload,
                    dst=1,
                    all_gather_group=tp_group,
                    all_gather_tensors=all_gather_map,
                )
                for h in handles:
                    h.wait()
            else:
                recvd, rhandles, postprocess = pp_group.irecv_tensor_dict(
                    src=0,
                    all_gather_group=tp_group,
                    all_gather_tensors=all_gather_map,
                )
                for h in rhandles:
                    h.wait()
                for fn in postprocess:
                    fn()
                assert recvd is not None
                torch.mcpu.synchronize()
                torch.testing.assert_close(
                    recvd["hidden_states"].cpu(), base_hidden.cpu()
                )
                torch.testing.assert_close(recvd["residual"].cpu(), base_residual.cpu())
                assert recvd["empty_field"].numel() == 0
                assert recvd["step_tag"] == 42

            # 3. Metadata-only / CPU-only fire-and-forget isend_tensor_dict reaping
            meta_only_payload = {
                "empty_dev": torch.empty((0,), dtype=torch.bfloat16, device="mcpu"),
                "cpu_scalar": torch.tensor(
                    [tp_group.rank_in_group + 9], dtype=torch.int64
                ),
                "step_note": "meta-only",
            }
            if pp_group.is_first_rank:
                first_handles = pp_group.isend_tensor_dict(meta_only_payload, dst=1)
                assert len(first_handles) == 1
                second_handles = pp_group.isend_tensor_dict(meta_only_payload, dst=1)
                assert first_handles[0].is_completed()
                for h in second_handles:
                    h.wait()
            else:
                for _ in range(2):
                    recvd_meta = pp_group.recv_tensor_dict(src=0)
                    assert recvd_meta is not None
                    assert recvd_meta["empty_dev"].numel() == 0
                    assert (
                        int(recvd_meta["cpu_scalar"].item())
                        == tp_group.rank_in_group + 9
                    )
                    assert recvd_meta["step_note"] == "meta-only"

            # 4. General group broadcast via MPI_Bcast + side-stream pp_broadcast
            tp_bcast = (
                torch.tensor([10.0, 20.0, 30.0], dtype=torch.bfloat16, device="mcpu")
                if tp_group.rank_in_group == 1
                else torch.zeros((3,), dtype=torch.bfloat16, device="mcpu")
            )
            tp_comm = tp_group.device_communicator
            assert isinstance(tp_comm, CpuMPICommunicator)
            tp_comm.broadcast(tp_bcast, src=1)
            torch.mcpu.synchronize()
            torch.testing.assert_close(
                tp_bcast.cpu(),
                torch.tensor([10.0, 20.0, 30.0], dtype=torch.bfloat16),
            )

            bcast_stream = torch.mcpu.Stream()
            expected_tokens = torch.tensor(
                [[11, 22, 33], [44, 55, 66]], dtype=torch.int64, device="mcpu"
            )
            bcast_buf = (
                expected_tokens.clone()
                if pp_group.is_last_rank
                else torch.zeros((2, 3), dtype=torch.int64, device="mcpu")
            )
            with torch.mcpu.stream(bcast_stream):
                pp_comm.pp_broadcast(bcast_buf, src=pp_group.world_size - 1)
            bcast_stream.synchronize()
            torch.testing.assert_close(bcast_buf.cpu(), expected_tokens.cpu())
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
