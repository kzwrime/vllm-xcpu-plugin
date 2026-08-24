# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_RESULT_PREFIX = "T4_QKV_RESULT="
_TIMEOUT_SECONDS = 120


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process_group(process: subprocess.Popen[str]) -> str:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
    return output


def _make_full_qkv(torch, seq_len: int):
    token_ids = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1, 1)
    head_ids = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1)
    base = ((token_ids % 4) * 16 + head_ids).expand(1, seq_len, 12, 64)
    return tuple((base + offset).to(torch.bfloat16) for offset in (0, 64, 128))


def _run_worker() -> None:
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    provided = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    if provided != MPI.THREAD_MULTIPLE:
        raise RuntimeError(
            "T4 QKV coverage requires MPI_THREAD_MULTIPLE, "
            f"but MPI provided thread level {provided}"
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if world_size != 2:
        raise RuntimeError(f"T4 QKV coverage requires world_size=2, got {world_size}")

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )

    import torch
    import torch_mcpu  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import (
        CpuMPICommunicator,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
    )
    vllm_config = VllmConfig(parallel_config=parallel_config)
    init_method = f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}"

    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=init_method,
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                pipeline_model_parallel_size=1,
                backend="gloo",
            )

            tp_group = get_tp_group()
            communicator = tp_group.device_communicator
            assert isinstance(communicator, CpuMPICommunicator)
            assert communicator.mpi_group_rank == rank
            assert communicator.mpi_group_size == world_size

            local_head_count = 12 // world_size
            assert local_head_count == 6
            head_start = rank * local_head_count

            for case, seq_len in (("fixed", 288), ("ragged", 840)):
                full_qkv = _make_full_qkv(torch, seq_len)
                gathered_qkv = []
                local_shape = None
                output_shape = None
                for full_tensor in full_qkv:
                    local_tensor = (
                        full_tensor.narrow(2, head_start, local_head_count)
                        .contiguous()
                        .to("mcpu")
                    )
                    gathered = communicator.all_gather(local_tensor, dim=2)
                    gathered_qkv.append(gathered)
                    local_shape = list(local_tensor.shape)
                    output_shape = list(gathered.shape)

                torch.accelerator.synchronize()
                for gathered, expected in zip(gathered_qkv, full_qkv, strict=True):
                    torch.testing.assert_close(
                        gathered.cpu(), expected, rtol=0, atol=0
                    )

                print(
                    _RESULT_PREFIX
                    + json.dumps(
                        {
                            "case": case,
                            "communicator": type(communicator).__name__,
                            "dtype": "bfloat16",
                            "exact_match": True,
                            "gather_dim": 2,
                            "input_shape": local_shape,
                            "output_shape": output_shape,
                            "rank": rank,
                            "rank_order": [0, 1],
                            "world_size": world_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()


def test_qwen35_tp2_vit_qkv_gather_matches_tp1() -> None:
    mpirun = shutil.which("mpirun")
    assert mpirun is not None, "T4 QKV coverage requires mpirun"

    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    pythonpath = [str(repo)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        pythonpath.append(existing_pythonpath)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_get_open_port()),
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "VLLM_CPU_USE_MPI": "1",
            "VLLM_PLUGINS": "xcpu_platform_plugin",
        }
    )
    command = [mpirun, "-np", "2", sys.executable, str(Path(__file__)), "--worker"]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        output = _stop_process_group(process)
        pytest.fail(
            f"T4 QKV MPI test timed out after {_TIMEOUT_SECONDS}s\n{output}"
        )

    assert process.returncode == 0, output
    records = [
        json.loads(line.removeprefix(_RESULT_PREFIX))
        for line in output.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(records) == 4, output

    expected_shapes = {
        "fixed": ([1, 288, 6, 64], [1, 288, 12, 64]),
        "ragged": ([1, 840, 6, 64], [1, 840, 12, 64]),
    }
    assert {(record["case"], record["rank"]) for record in records} == {
        (case, rank) for case in expected_shapes for rank in (0, 1)
    }
    for record in sorted(records, key=lambda item: (item["case"], item["rank"])):
        expected_input, expected_output = expected_shapes[record["case"]]
        assert record == {
            "case": record["case"],
            "communicator": "CpuMPICommunicator",
            "dtype": "bfloat16",
            "exact_match": True,
            "gather_dim": 2,
            "input_shape": expected_input,
            "output_shape": expected_output,
            "rank": record["rank"],
            "rank_order": [0, 1],
            "world_size": 2,
        }
        print(_RESULT_PREFIX + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("usage: test_qwen35_vit_qkv_mpi.py --worker")
    _run_worker()
