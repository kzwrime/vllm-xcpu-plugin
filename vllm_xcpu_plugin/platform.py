from typing import TYPE_CHECKING

from vllm.platforms.cpu import CpuPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

import vllm_xcpu_plugin.envs as envs_xcpu


class XcpuPlatform(CpuPlatform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        if envs_xcpu.VLLM_CPU_USE_MPI:
            return (
                "vllm_xcpu_plugin.distributed.cpu_mpi_communicator.CpuMPICommunicator"  # noqa
            )
        else:
            return (
                "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"  # noqa
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().check_and_update_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "vllm.v1.worker.cpu_worker.CPUWorker":
            parallel_config.worker_cls = "vllm_xcpu_plugin.worker.worker_v1.XcpuWorker"

        assert parallel_config.worker_cls == (
            "vllm_xcpu_plugin.worker.worker_v1.XcpuWorker"
        )
