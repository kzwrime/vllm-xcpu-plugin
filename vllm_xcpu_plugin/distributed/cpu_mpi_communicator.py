import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.all2all import (
    AgRsAll2AllManager,
    All2AllManagerBase,
)
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import logger

MPI_ALLTOALLV_BACKENDS = {
    "mpi_alltoallv",
    "mpi_alltoallv_v1",
    "mpi_alltoallv_v2",
    "mpi_alltoallv_v3",
    "mpi_alltoallv_v4",
    "mpi_alltoallv_v5",
    "mpi_alltoallv_v6",
}


class CpuMPICommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        import mpi4py.rc

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

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

        assert self.mpi_group_rank == self.rank, f"{self.mpi_group_rank}, {self.rank}"
        assert self.mpi_group_size == self.world_size, (
            f"{self.mpi_group_size}, {self.world_size}"
        )

        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "all_to_all_single":
                from vllm_xcpu_plugin.distributed.all2all import (
                    All2allvSingleAll2AllManager,
                )

                self.all2all_manager = All2allvSingleAll2AllManager(
                    cpu_group=self.cpu_group
                )
            elif (
                self.all2all_backend == "torch_all_to_all_single"
                or self.all2all_backend in MPI_ALLTOALLV_BACKENDS
            ):
                # Custom Prepare/Finalize owns dispatch/combine. The base
                # manager supplies only rank/world-size control-plane state.
                self.all2all_manager = All2AllManagerBase(cpu_group=self.cpu_group)
            else:
                raise ValueError(
                    f"Unknown/Unsupported all2all backend: {self.all2all_backend}"
                )
            logger.info(
                "MoE all2all backend=%s manager=%s",
                self.all2all_backend,
                type(self.all2all_manager).__name__,
            )

        self.comm_ptr = self.mpi_group_comm.py2f()
        self.comm_ptr_wrapper = torch.tensor(
            [self.comm_ptr], dtype=torch.int64, device="cpu"
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        import torch_mpi_ext

        # logger.info(f"all_reduce rank: {self.mpi_group_rank}, "
        #     f"input_.shape: {input_.shape}, input_.dtype: {input_.dtype}")
        torch_mpi_ext.ops.all_reduce__wrapper(input_, self.comm_ptr_wrapper)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        from torch_xcpu import ops as xcpu_ops

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

        xcpu_ops.all_gather_into_tensor_out_v2(
            output_tensor, input_, self.comm_ptr_wrapper, dim=dim
        )

        return output_tensor

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(input_, list):
            output_list: list[torch.Tensor] = []
            for tensor in input_:
                output = self.all_gatherv(tensor, dim=dim, sizes=sizes)
                assert isinstance(output, torch.Tensor)
                output_list.append(output)
            return output_list

        if not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"invalid dim {dim} for input shape {tuple(input_.shape)}")
        if dim < 0:
            dim += input_.dim()
        if sizes is None:
            sizes = [input_.size(dim)] * self.world_size
        if len(sizes) != self.world_size or any(size < 0 for size in sizes):
            raise ValueError("sizes must contain one non-negative value per rank")
        if input_.size(dim) != sizes[self.rank_in_group]:
            raise ValueError(
                "local input size does not match sizes for this rank: "
                f"{input_.size(dim)} != {sizes[self.rank_in_group]}"
            )
        if self.world_size == 1:
            return input_
        from torch_xcpu import ops as xcpu_ops

        output_shape = list(input_.shape)
        output_shape[dim] = sum(sizes)
        output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        sizes_tensor = torch.tensor(sizes, dtype=torch.int64, device="cpu")
        xcpu_ops.all_gatherv_into_tensor_out_v2(
            output, input_, sizes_tensor, self.comm_ptr_wrapper, dim
        )
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"invalid dim {dim} for input shape {tuple(input_.shape)}")
        if dim < 0:
            dim += input_.dim()
        if input_.size(dim) % self.world_size != 0:
            raise ValueError(
                "input size along dim must be divisible by world size: "
                f"{input_.size(dim)} % {self.world_size} != 0"
            )
        if self.world_size == 1:
            return input_

        import torch_mpi_ext

        output_shape = list(input_.shape)
        output_shape[dim] //= self.world_size
        output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        torch_mpi_ext.ops.reduce_scatter_out_wrapper(
            output, input_, self.comm_ptr_wrapper, dim
        )
        return output

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"invalid dim {dim} for input shape {tuple(input_.shape)}")
        if dim < 0:
            dim += input_.dim()
        if sizes is None:
            return self.reduce_scatter(input_, dim=dim)
        if len(sizes) != self.world_size or any(size < 0 for size in sizes):
            raise ValueError("sizes must contain one non-negative value per rank")
        if input_.size(dim) != sum(sizes):
            raise ValueError(
                "input size along dim must equal sum(sizes): "
                f"{input_.size(dim)} != {sum(sizes)}"
            )
        if self.world_size == 1:
            return input_
        if len(set(sizes)) == 1:
            return self.reduce_scatter(input_, dim=dim)

        import torch_mpi_ext

        output_shape = list(input_.shape)
        output_shape[dim] = sizes[self.rank_in_group]
        output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        sizes_tensor = torch.tensor(sizes, dtype=torch.int64, device="cpu")
        torch_mpi_ext.ops.reduce_scatterv_out_wrapper(
            output, input_, sizes_tensor, self.comm_ptr_wrapper, dim
        )
        return output

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

    def destroy(self):
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None  # type: ignore[has-type]

    pass
