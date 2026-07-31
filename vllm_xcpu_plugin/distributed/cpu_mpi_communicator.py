import importlib
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.all2all import (
    All2AllManagerBase,
)
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import logger

import vllm_xcpu_plugin.envs as envs_xcpu

_all2all_module = importlib.import_module(
    "vllm.distributed.device_communicators.all2all"
)
NaiveAll2AllManager = cast(Any, getattr(_all2all_module, "NaiveAll2AllManager", None))
if NaiveAll2AllManager is None:
    NaiveAll2AllManager = cast(Any, _all2all_module.AgRsAll2AllManager)


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
            self.all2all_backend = envs_xcpu.VLLM_ALL2ALL_BACKEND_XCPU
            if self.all2all_backend == "allgather_reducescatter":  # type: ignore[has-type]
                logger.warning(
                    "Not supported all2all backend %s, fallback to all_to_all_single",
                    self.all2all_backend,
                )
                self.all2all_backend = "all_to_all_single"  # type: ignore[assignment]
            if self.all2all_backend == "naive":  # type: ignore[has-type]
                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "all_to_all_single":  # type: ignore[has-type]
                from vllm_xcpu_plugin.distributed.all2all import (
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
        self.comm_ptr_wrapper = torch.tensor([self.comm_ptr])

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
        if max(sizes) == 0:
            return input_
        if len(set(sizes)) == 1:
            return self.all_gather(input_, dim=dim)

        max_size = max(sizes)
        if input_.size(dim) < max_size:
            pad_shape = list(input_.shape)
            pad_shape[dim] = max_size - input_.size(dim)
            padding = torch.zeros(pad_shape, dtype=input_.dtype, device=input_.device)
            padded = torch.cat((input_, padding), dim=dim)
        else:
            padded = input_
        gathered = self.all_gather(padded, dim=dim)
        rank_chunks = gathered.split(max_size, dim=dim)
        return torch.cat(
            [
                chunk.narrow(dim, 0, size)
                for chunk, size in zip(rank_chunks, sizes, strict=True)
            ],
            dim=dim,
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

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
