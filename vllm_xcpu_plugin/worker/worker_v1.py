from vllm.logger import logger
from vllm.platforms import current_platform
from vllm.v1.worker.cpu_worker import CPUWorker

import vllm_xcpu_plugin.envs as envs_xcpu


class XcpuWorker(CPUWorker):
    def init_device(self):
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

        # Call init_worker_distributed_environment、CPUModelRunner...
        super().init_device()
