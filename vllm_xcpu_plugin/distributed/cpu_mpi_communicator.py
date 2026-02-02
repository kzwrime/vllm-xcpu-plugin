import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.all2all import (
    All2AllManagerBase,
    NaiveAll2AllManager,
)
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import logger

import vllm_xcpu_plugin.envs as envs_xcpu

try:
    import mpi4py.rc

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py not found.") from None

try:
    import torch_mpi_ext
except ImportError:
    raise ImportError("torch_mpi_ext not found.") from None


class CpuMPICommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        logger.info("CpuMPICommunicator initializing ...")

        assert MPI.Is_initialized()

        num_ranks = cpu_group.size()
        assert num_ranks > 0
        logger.info("num_ranks: %d", num_ranks)

        mpi_global_rank = MPI.COMM_WORLD.Get_rank()

        global_rank_tensor = torch.tensor([mpi_global_rank], dtype=torch.int32)

        group_ranks_ = torch.zeros(num_ranks, dtype=torch.int32)
        dist.all_gather_into_tensor(
            group_ranks_, global_rank_tensor, group=self.cpu_group
        )
        group_ranks = group_ranks_.tolist()
        logger.info("[%d] group_ranks: %s", self.global_rank, str(group_ranks))
        # mpi_group = MPI.COMM_WORLD.group.Incl(group_ranks)
        # self.mpi_group_comm = MPI.Intracomm.Create_from_group(mpi_group)

        min_rank_in_group_ = torch.tensor([mpi_global_rank], dtype=torch.int32)

        dist.all_reduce(min_rank_in_group_, op=dist.ReduceOp.MIN, group=self.cpu_group)
        min_rank_in_group = int(min_rank_in_group_.item())
        logger.info("[%d] min_rank_in_group: %d", mpi_global_rank, min_rank_in_group)
        self.mpi_group_comm = MPI.COMM_WORLD.Split(min_rank_in_group)
        group_ranks_verify = torch.zeros(num_ranks, dtype=torch.int32)
        self.mpi_group_comm.Allgather(global_rank_tensor, group_ranks_verify)
        logger.info(
            "[%d] group_ranks_verify: %s",
            mpi_global_rank,
            str(group_ranks_verify.tolist()),
        )

        self.mpi_group_rank = self.mpi_group_comm.Get_rank()
        self.mpi_group_size = self.mpi_group_comm.Get_size()
        logger.info(
            "CpuMPICommunicator initialized, rank: %d, world_size: %d",
            self.mpi_group_rank,
            self.mpi_group_size,
        )

        assert self.mpi_group_rank == self.rank
        assert self.mpi_group_size == self.world_size

        if self.use_all2all:
            self.all2all_backend = envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU
            if self.all2all_backend == "naive":  # type: ignore[has-type]
                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "all_to_all_single":  # type: ignore[has-type]
                from vllm.distributed.device_communicators.all2all import (
                    All2allvSingleAll2AllManager,
                )

                self.all2all_manager = All2allvSingleAll2AllManager(
                    cpu_group=self.cpu_group
                )
            elif self.all2all_backend in ("torch_all_to_all_single", "mpi_alltoallv"):  # type: ignore[has-type]
                # do nothing
                self.all2all_manager = All2AllManagerBase(cpu_group=self.cpu_group)
            else:
                raise ValueError(
                    f"Unknown/Unsupported all2all backend: {self.all2all_backend}"
                )
            logger.info("Using all2all_backend = %s", self.all2all_backend)

        self.comm_ptr = self.mpi_group_comm.py2f()

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # logger.info(f"all_reduce rank: {self.mpi_group_rank}, "
        #     f"input_.shape: {input_.shape}, input_.dtype: {input_.dtype}")
        torch_mpi_ext.ops.all_reduce_(input_, self.comm_ptr)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # logger.info(f"all_gather rank: {self.mpi_group_rank}, "
        #     f"input_.shape: {input_.shape}, input_.dtype: {input_.dtype}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )

        torch_mpi_ext.ops.all_gather_into_tensor_out(
            output_tensor, input_, self.comm_ptr, dim=dim
        )

        return output_tensor

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        raise NotImplementedError

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        hidden_states, router_logits = self.all2all_manager.dispatch(
            hidden_states, router_logits, is_sequence_parallel
        )
        return hidden_states, router_logits

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(
            hidden_states, is_sequence_parallel
        )
        return hidden_states

    def destroy(self):
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None

    pass
