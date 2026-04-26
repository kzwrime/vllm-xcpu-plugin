# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
from typing import Any

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_xcpu_plugin.envs as envs_xcpu

from .model_runner import McpuModelRunner, McpuModelRunnerV2

logger = init_logger(__name__)


class McpuWorker(Worker):
    """A mcpu worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        import torch_mcpu  # noqa

        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        print(f"device_config: {device_config}")
        assert device_config.device_type == "privateuseone"

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            world_rank = (
                self.parallel_config.data_parallel_rank
                * self.parallel_config.world_size
                + rank
            )
            worker_name = (
                f"{vllm_config.instance_id}-world-rank-{world_rank}-rank-{self.rank}"
            )
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "PrivateUse1"],
            )

            def _eplb_on_profiler_stop() -> None:
                eplb = getattr(getattr(self, "model_runner", None), "eplb_state", None)
                if eplb is not None:
                    logger.info("Profiler stopped, dumping EPLB statistics window.")
                    eplb.log_all_statistics(is_profiler_stop=True)

            self.profiler.add_stop_callback(_eplb_on_profiler_stop)

    def init_device(self):
        import torch_mcpu  # noqa

        world_rank_across_dp = (
            self.parallel_config.data_parallel_rank * self.parallel_config.world_size
        ) + self.rank
        world_size_across_dp = self.parallel_config.world_size_across_dp

        logger.info(
            (
                "rank: %d, local_rank: %d, world_size: %d, dist_backend: %s, "
                "self.distributed_init_method: %s, "
                "world_rank_across_dp: %d, world_size_across_dp: %d"
            ),
            self.rank,
            self.local_rank,
            self.parallel_config.world_size,
            current_platform.dist_backend,
            self.distributed_init_method,
            world_rank_across_dp,
            world_size_across_dp,
        )

        if envs_xcpu.VLLM_CPU_USE_MPI:
            import mpi4py.rc

            mpi4py.rc.initialize = False
            mpi4py.rc.finalize = False
            from mpi4py import MPI

            if not MPI.Is_initialized():
                MPI.Init()
            self.mpi_finalize = MPI.Finalize
            self.mpi_initialized = True
            self.mpi_world_comm = MPI.COMM_WORLD

            # mpi_rank = self.mpi_world_comm.Get_rank()
            # mpi_size = self.mpi_world_comm.Get_size()

            """
            Warning: In DP + Dense Model, a global gloo communication group 
              may not be established; instead, DP-Size gloo communication 
              groups will be established.
            vllm/distributed/parallel_state.py:init_distributed_environment()
              may not adjust the distributed_init_method
            """
            # assert mpi_rank == world_rank_across_dp, (
            #     f"mpi_rank: {mpi_rank} != "
            #     f"global_world_rank: {world_rank_across_dp}")
            # assert mpi_size == world_size_across_dp, (
            #     f"mpi_world_size: {mpi_size} != "
            #     f"world_size_across_dp: {world_size_across_dp}")

            import socket

            host_name = socket.gethostname()
            host_ip = socket.gethostbyname(host_name)
            logger.info(
                "rank: %d, %s@%s, MPI.Is_initialized(): %d",
                self.rank,
                host_name,
                host_ip,
                MPI.Is_initialized(),
            )
            if envs.VLLM_EPLB_COMM_BACKEND == "mpi":
                mpi_rank = self.mpi_world_comm.Get_rank()
                mpi_size = self.mpi_world_comm.Get_size()
                assert mpi_rank == world_rank_across_dp, (
                    f"MPI rank mismatch for EPLB: mpi_rank={mpi_rank}, "
                    f"torch_world_rank={world_rank_across_dp}"
                )
                assert mpi_size == world_size_across_dp, (
                    f"MPI world size mismatch for EPLB: mpi_size={mpi_size}, "
                    f"torch_world_size={world_size_across_dp}"
                )
                logger.info(
                    "EPLB MPI backend verified: mpi_rank=%d world_size=%d",
                    mpi_rank,
                    mpi_size,
                )
        elif envs.VLLM_EPLB_COMM_BACKEND == "mpi":
            raise RuntimeError(
                "VLLM_EPLB_COMM_BACKEND=mpi requires VLLM_CPU_USE_MPI=1."
            )

        # device = self.device_config.device
        self.device = torch.device("mcpu:0")
        torch.accelerator.set_device_index(self.device)
        torch.accelerator.empty_cache()
        # self.init_gpu_memory = torch.xpu.get_device_properties(
        #     self.local_rank
        # ).total_memory

        # ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        # ENV_LOCAL_WORLD_SIZE = os.getenv(
        #     "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        # )
        # os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        # os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        # os.environ["LOCAL_RANK"] = str(self.local_rank)

        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # global all_reduce needed for overall oneccl warm up
        # torch.distributed.all_reduce(torch.zeros(1).xpu())

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        # kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        # assert kv_cache_space is not None
        # self.requested_memory = kv_cache_space * GiB_bytes
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        # logger.debug(
        #     "worker requested memory: %sGiB", format_gib(self.requested_memory)
        # )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        model_runner = (
            McpuModelRunnerV2 if self.use_v2_model_runner else McpuModelRunner
        )
        self.model_runner = model_runner(  # type: ignore
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def determine_available_memory(self) -> int:
        available_memory = super().determine_available_memory()

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        if kv_cache_space is None:
            return available_memory

        kv_cache_space_bytes = kv_cache_space * GiB_bytes
        logger.info(
            "Force reset available kv cache memory from %sGiB to "
            "VLLM_CPU_KVCACHE_SPACE: %sGiB",
            format_gib(available_memory),
            format_gib(kv_cache_space_bytes),
        )
        self.available_kv_cache_memory_bytes = kv_cache_space_bytes
        return kv_cache_space_bytes

    def shutdown(self):
        return
