from typing import Any

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
    "mpi_alltoallv_v7",
    "mpi_alltoallv_v8",
}


class _MpiRequestHandle:
    """Handle wrapping an mpi4py non-blocking request with payload retention."""

    reliable_is_completed = True

    def __init__(self, req: Any, keep_alive: Any = None) -> None:
        self._req = req
        self._keep_alive = keep_alive
        self._done = False

    def wait(self) -> None:
        if not self._done:
            self._req.Wait()
            self._done = True
            self._keep_alive = None

    def is_completed(self) -> bool:
        if self._done:
            return True
        if self._req.Test():
            self._done = True
            self._keep_alive = None
            return True
        return False


class _McpuEventHandle:
    """Handle backed by an MCPU stream event for P2P completion and lifetime."""

    def __init__(
        self,
        event: Any,
        wait_stream: Any = None,
        keep_alive: Any = None,
    ) -> None:
        self._event = event
        self._wait_stream = wait_stream
        self._keep_alive = keep_alive
        self._done = False

    def wait(self) -> None:
        if not self._done:
            if self._wait_stream is not None:
                self._wait_stream.wait_event(self._event)
            else:
                self._event.synchronize()
            self._done = True
            self._keep_alive = None

    def is_completed(self) -> bool:
        if self._done:
            return True
        if self._event.query():
            self._done = True
            self._keep_alive = None
            return True
        return False


class CpuMPICommunicator(DeviceCommunicatorBase):
    supports_custom_tensor_dict = True
    supports_custom_pp_broadcast = True

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        import mpi4py.rc

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

        logger.info("CpuMPICommunicator initializing ...")

        assert MPI.Is_initialized()

        num_ranks = cpu_group.size()
        assert num_ranks > 0
        logger.info("num_ranks: %d", num_ranks)

        from vllm_xcpu_plugin.distributed.mpi_world import require_mpi_world

        mpi_world = require_mpi_world()
        cluster_world_comm = mpi_world.cluster_comm
        mpi_cluster_rank = cluster_world_comm.Get_rank()

        cluster_rank_tensor = torch.tensor([mpi_cluster_rank], dtype=torch.int32)

        group_ranks_ = torch.zeros(num_ranks, dtype=torch.int32)
        dist.all_gather_into_tensor(
            group_ranks_, cluster_rank_tensor, group=self.cpu_group
        )
        group_ranks = group_ranks_.tolist()
        logger.info("[%d] group_ranks: %s", self.global_rank, str(group_ranks))
        # mpi_group = MPI.COMM_WORLD.group.Incl(group_ranks)
        # self.mpi_group_comm = MPI.Intracomm.Create_from_group(mpi_group)

        min_rank_in_group_ = torch.tensor([mpi_cluster_rank], dtype=torch.int32)

        dist.all_reduce(min_rank_in_group_, op=dist.ReduceOp.MIN, group=self.cpu_group)
        min_rank_in_group = int(min_rank_in_group_.item())
        logger.info("[%d] min_rank_in_group: %d", mpi_cluster_rank, min_rank_in_group)
        # Key by group rank so MPI group rank order matches cpu_group exactly.
        self.mpi_group_comm = cluster_world_comm.Split(min_rank_in_group, self.rank)
        group_ranks_verify = torch.zeros(num_ranks, dtype=torch.int32)
        self.mpi_group_comm.Allgather(cluster_rank_tensor, group_ranks_verify)
        logger.info(
            "[%d] group_ranks_verify: %s",
            mpi_cluster_rank,
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
            if self.all2all_backend in ("mpi_alltoallv_v7", "mpi_alltoallv_v8"):
                import vllm_xcpu_plugin.envs as envs_xcpu

                if not envs_xcpu.VLLM_XCPU_ENABLE_AF_EP:
                    raise ValueError(f"{self.all2all_backend} requires AF-EP")
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
        # Isolated communicators so host metadata, tensor P2P, and side-stream
        # PP sampled-token relay never collide on tags or MPI worker threads.
        self.mpi_meta_comm = self.mpi_group_comm.Dup()
        self.mpi_bcast_comm = self.mpi_group_comm.Dup()
        self.bcast_comm_ptr_wrapper = torch.tensor(
            [self.mpi_bcast_comm.py2f()], dtype=torch.int64, device="cpu"
        )
        self._p2p_send_stream: Any = None
        self._p2p_recv_stream: Any = None
        self._pending_meta_handles: list[_MpiRequestHandle] = []

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        import torch_mpi_ext

        import vllm_xcpu_plugin.envs as envs_xcpu

        if envs_xcpu.VLLM_XCPU_ALLREDUCE_INPLACE:
            torch_mpi_ext.ops.all_reduce__wrapper(input_, self.comm_ptr_wrapper)
            return input_

        # Keep allocation visible to Inductor so it can plan/reuse the output
        # without functionalizing a mutation of the input (which may be a view).
        output = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
        torch_mpi_ext.ops.all_reduce_out_wrapper(
            output, input_, self.comm_ptr_wrapper
        )
        return output

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

    def _normalize_device(self, dev: torch.device | str) -> torch.device:
        device = torch.device(dev)
        if device.type in ("mcpu", "privateuseone"):
            return torch.device("mcpu", torch.mcpu.current_device())
        return device

    def _get_p2p_send_stream(self) -> Any:
        if self._p2p_send_stream is None:
            self._p2p_send_stream = torch.mcpu.Stream()
        return self._p2p_send_stream

    def _get_p2p_recv_stream(self) -> Any:
        if self._p2p_recv_stream is None:
            self._p2p_recv_stream = torch.mcpu.Stream()
        return self._p2p_recv_stream

    @staticmethod
    def _should_use_all_gather(
        key: str,
        numel: int,
        all_gather_group: Any | None,
        all_gather_tensors: dict[str, bool] | None,
    ) -> bool:
        if all_gather_group is None:
            return False
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from group-local src rank via MPI_Bcast."""
        if self.world_size <= 1:
            return tensor
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        import torch_mpi_ext

        contig = tensor if tensor.is_contiguous() else tensor.contiguous()
        torch_mpi_ext.ops.broadcast_(contig, src, self.comm_ptr_wrapper)
        if contig is not tensor:
            tensor.copy_(contig)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Send a tensor to destination rank using MPI_Send."""
        import torch_mpi_ext

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert 0 <= dst < self.world_size, f"Invalid dst rank ({dst})"
        contig = tensor.contiguous()
        torch_mpi_ext.ops.send(contig, dst, 0, self.comm_ptr_wrapper)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receive a tensor from source rank using MPI_Recv."""
        import torch_mpi_ext

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        tensor = torch.empty(
            size, dtype=dtype, device=self._normalize_device(self.device)
        )
        torch_mpi_ext.ops.recv_(tensor, src, 0, self.comm_ptr_wrapper)
        return tensor

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
        all_gather_group: Any | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Any]:
        if self.world_size <= 1:
            return []
        import torch_mpi_ext
        from vllm.distributed.parallel_state import _split_tensor_dict

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert 0 <= dst < self.world_size, f"Invalid dst rank ({dst})"

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)

        self._pending_meta_handles = [
            h for h in self._pending_meta_handles if not h.is_completed()
        ]
        meta_req = self.mpi_meta_comm.isend(metadata_list, dest=dst, tag=0)
        meta_handle = _MpiRequestHandle(meta_req, keep_alive=metadata_list)
        self._pending_meta_handles.append(meta_handle)
        handles: list[Any] = [meta_handle]

        dev_sends: list[tuple[torch.Tensor, int]] = []
        cpu_sends: list[tuple[torch.Tensor, int]] = []
        tensor_tag = 0
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue
            tensor_tag += 1
            if self._should_use_all_gather(
                key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
            contig = tensor.contiguous()
            if contig.is_cpu:
                cpu_sends.append((contig, tensor_tag))
            else:
                dev_sends.append((contig, tensor_tag))

        if dev_sends:
            main_stream = torch.mcpu.current_stream()
            send_stream = self._get_p2p_send_stream()
            send_stream.wait_stream(main_stream)
            with torch.mcpu.stream(send_stream):
                for contig, tag in dev_sends:
                    torch_mpi_ext.ops.send(contig, dst, tag, self.comm_ptr_wrapper)
            send_event = send_stream.record_event()
            handles.append(
                _McpuEventHandle(
                    send_event,
                    wait_stream=None,
                    keep_alive=[t for t, _ in dev_sends],
                )
            )

        for contig, tag in cpu_sends:
            torch_mpi_ext.ops.send(contig, dst, tag, self.comm_ptr_wrapper)

        return handles

    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: Any | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[dict[str, Any] | None, list[Any], list[Any]]:
        if self.world_size <= 1:
            return None, [], []
        import torch_mpi_ext
        from vllm.distributed.parallel_state import TensorMetadata

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        recv_metadata_list = self.mpi_meta_comm.recv(source=src, tag=0)
        tensor_dict: dict[str, Any] = {}
        handles: list[Any] = []
        postprocess: list[Any] = []

        dev_recvs: list[tuple[torch.Tensor, int]] = []
        cpu_recvs: list[tuple[torch.Tensor, int]] = []
        tensor_tag = 0
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                target_device = self._normalize_device(value.device)
                full_tensor = torch.empty(
                    value.size, dtype=value.dtype, device=target_device
                )
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue
                tensor_tag += 1
                if self._should_use_all_gather(
                    key, full_tensor.numel(), all_gather_group, all_gather_tensors
                ):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                        all_gather_rank
                    ]
                    recv_target = slice_tensor

                    def _postprocess(
                        key: str = key,
                        slice_tensor: torch.Tensor = slice_tensor,
                        orig_shape: tuple[int, ...] = tuple(orig_shape),
                        all_gather_group: Any = all_gather_group,
                    ) -> None:
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(
                            slice_tensor, dim=0
                        ).reshape(orig_shape)

                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    recv_target = full_tensor
                    tensor_dict[key] = full_tensor

                if recv_target.is_cpu:
                    cpu_recvs.append((recv_target, tensor_tag))
                else:
                    dev_recvs.append((recv_target, tensor_tag))
            else:
                tensor_dict[key] = value

        if dev_recvs:
            main_stream = torch.mcpu.current_stream()
            recv_stream = self._get_p2p_recv_stream()
            recv_stream.wait_stream(main_stream)
            with torch.mcpu.stream(recv_stream):
                for recv_target, tag in dev_recvs:
                    torch_mpi_ext.ops.recv_(
                        recv_target, src, tag, self.comm_ptr_wrapper
                    )
            recv_event = recv_stream.record_event()
            handles.append(
                _McpuEventHandle(
                    recv_event,
                    wait_stream=main_stream,
                    keep_alive=[t for t, _ in dev_recvs],
                )
            )

        for recv_target, tag in cpu_recvs:
            torch_mpi_ext.ops.recv_(recv_target, src, tag, self.comm_ptr_wrapper)

        return tensor_dict, handles, postprocess

    def pp_broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        """Broadcast PP sampled tokens via formal MPI send/recv."""
        if self.world_size <= 1:
            return
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        import torch_mpi_ext

        contig = tensor if tensor.is_contiguous() else tensor.contiguous()
        if self.rank_in_group == src:
            for peer in range(self.world_size):
                if peer != src:
                    torch_mpi_ext.ops.send(
                        contig, peer, peer, self.bcast_comm_ptr_wrapper
                    )
        else:
            torch_mpi_ext.ops.recv_(
                contig, src, self.rank_in_group, self.bcast_comm_ptr_wrapper
            )
            if contig is not tensor:
                tensor.copy_(contig)

    def destroy(self):
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None  # type: ignore[has-type]
        for stream_attr in ("_p2p_send_stream", "_p2p_recv_stream"):
            stream = getattr(self, stream_attr, None)
            if stream is not None:
                stream.synchronize()
                setattr(self, stream_attr, None)
        from mpi4py import MPI

        if MPI.Is_initialized() and not MPI.Is_finalized():
            for meta_handle in getattr(self, "_pending_meta_handles", ()):
                meta_handle.wait()
            self._pending_meta_handles = []
            for attr in ("mpi_meta_comm", "mpi_bcast_comm", "mpi_group_comm"):
                comm = getattr(self, attr, None)
                if comm is not None:
                    comm.Free()
                    setattr(self, attr, None)

    pass
