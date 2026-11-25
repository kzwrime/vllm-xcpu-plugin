# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MPI EPLB communicator backed by torch_mpi_ext.ops.batch_isend_irecv."""

from __future__ import annotations

import numpy as np
import torch
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.eplb_utils import device_stream
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_xcpu_plugin.distributed.cpu_mpi_communicator import CpuMPICommunicator

_MAX_TENSORS_PER_EXPERT = 8
_MAX_SEQ_PER_EXPERT = 4
_EXPERT_STRIDE = _MAX_SEQ_PER_EXPERT * _MAX_TENSORS_PER_EXPERT
_LAYER_EXPERT_CAPACITY = 1024
_LAYER_STRIDE = _LAYER_EXPERT_CAPACITY * _EXPERT_STRIDE


def _get_mpi():
    # Defer mpi4py import until a worker uses MPI so CLI/headless launchers
    # under mpirun do not trigger MPI_Init() and deadlock OpenMPI PMIx.
    import mpi4py.rc

    mpi4py.rc(initialize=False, finalize=False)
    from mpi4py import MPI

    return MPI


class MpiEplbCommunicator(EplbCommunicator):
    """EPLB P2P communicator using batch_isend_irecv over the EPLB MPI group."""

    def __init__(self, group_coordinator: GroupCoordinator) -> None:
        super().__init__()
        self._ep_group = group_coordinator
        self._world_size = group_coordinator.world_size
        self._rank = group_coordinator.rank_in_group
        self._layer_idx = 0
        self._stream: torch.Stream | None = None
        self._sends: list[tuple[torch.Tensor, int, int]] = []
        self._recvs: list[tuple[torch.Tensor, int, int]] = []
        self._pair_seq: dict[tuple[str, int, int], int] = {}
        self._used_peer_tags: dict[tuple[str, int], set[int]] = {}

        if self._world_size > 1:
            device_comm = group_coordinator.device_communicator
            if not isinstance(device_comm, CpuMPICommunicator):
                raise RuntimeError(
                    "MpiEplbCommunicator requires VLLM_CPU_USE_MPI=1 and "
                    "CpuMPICommunicator on the EPLB process group"
                )
            mpi_comm = device_comm.mpi_group_comm
            if mpi_comm.Get_rank() != self._rank:
                raise RuntimeError(
                    f"EPLB MPI rank mismatch: mpi={mpi_comm.Get_rank()}, "
                    f"group={self._rank}"
                )
            if mpi_comm.Get_size() != self._world_size:
                raise RuntimeError(
                    f"EPLB MPI size mismatch: mpi={mpi_comm.Get_size()}, "
                    f"group={self._world_size}"
                )
            self._comm_ptr_wrapper: torch.Tensor | None = (
                device_comm.comm_ptr_wrapper
            )
            MPI = _get_mpi()
            tag_ub = mpi_comm.Get_attr(MPI.TAG_UB)
            self._tag_ub = (
                int(tag_ub) if tag_ub is not None and int(tag_ub) > 0 else 32767
            )
        else:
            self._comm_ptr_wrapper = None
            self._tag_ub = 32767

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False

    def set_stream(self, stream: torch.Stream | None) -> None:
        self._stream = stream

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int) -> None:
        del old_indices
        self._layer_idx = int(layer_idx)

    def _next_tags(
        self, op: str, peer_rank: int, expert_id: int, num_tensors: int
    ) -> list[int]:
        if peer_rank < 0 or peer_rank >= self._world_size:
            raise ValueError(
                f"Invalid peer rank {peer_rank} for EPLB world_size={self._world_size}"
            )
        if expert_id < 0:
            raise ValueError(f"Invalid negative expert_id {expert_id}")
        if num_tensors == 0:
            return []
        if num_tensors > _MAX_TENSORS_PER_EXPERT:
            raise ValueError(
                f"num_tensors={num_tensors} exceeds MpiEplbCommunicator limit "
                f"({_MAX_TENSORS_PER_EXPERT})"
            )

        key = (op, int(peer_rank), int(expert_id))
        seq = self._pair_seq.get(key, 0)
        if seq >= _MAX_SEQ_PER_EXPERT:
            raise ValueError(
                f"Repeated {op} count {seq + 1} for peer={peer_rank}, "
                f"expert={expert_id} exceeds limit ({_MAX_SEQ_PER_EXPERT})"
            )
        self._pair_seq[key] = seq + 1

        # Keep tag deterministic across sender/receiver within [0, _tag_ub].
        # Only include layer_idx when TAG_UB exceeds one 1024-expert layer block
        # so MPI_TAG_UB == 32767 still supports up to 1024 experts collision-free.
        modulo = self._tag_ub + 1
        layer_offset = (
            self._layer_idx * _LAYER_STRIDE if modulo >= 2 * _LAYER_STRIDE else 0
        )
        base = (
            layer_offset
            + (int(expert_id) * _EXPERT_STRIDE)
            + (seq * _MAX_TENSORS_PER_EXPERT)
        )
        tags = [(base + tensor_idx) % modulo for tensor_idx in range(num_tensors)]

        peer_used = self._used_peer_tags.setdefault((op, int(peer_rank)), set())
        if peer_used.intersection(tags):
            raise RuntimeError(
                f"MPI EPLB tag collision within execute() batch for op={op}, "
                f"peer={peer_rank}, expert_id={expert_id}, tag_ub={self._tag_ub}"
            )
        peer_used.update(tags)
        return tags

    def add_send(
        self, tensors: list[torch.Tensor], dst_rank: int, expert_id: int
    ) -> None:
        tags = self._next_tags("send", dst_rank, expert_id, len(tensors))
        for tensor, tag in zip(tensors, tags):
            if not tensor.is_contiguous():
                raise ValueError(
                    "MpiEplbCommunicator.add_send requires contiguous tensors"
                )
            self._sends.append((tensor, int(dst_rank), tag))

    def add_recv(
        self, tensors: list[torch.Tensor], src_rank: int, expert_id: int
    ) -> None:
        tags = self._next_tags("recv", src_rank, expert_id, len(tensors))
        for tensor, tag in zip(tensors, tags):
            if not tensor.is_contiguous():
                raise ValueError(
                    "MpiEplbCommunicator.add_recv requires contiguous tensors"
                )
            self._recvs.append((tensor, int(src_rank), tag))

    def execute(self) -> None:
        if not self._sends and not self._recvs:
            self._pair_seq.clear()
            self._used_peer_tags.clear()
            return
        if self._comm_ptr_wrapper is None:
            raise RuntimeError(
                "MpiEplbCommunicator.execute requires world_size > 1 and "
                "active MPI comm"
            )
        if self._stream is not None:
            MPI = _get_mpi()
            thread_level = MPI.Query_thread()
            if thread_level != MPI.THREAD_MULTIPLE:
                raise RuntimeError(
                    "Async EPLB over MPI requires MPI_THREAD_MULTIPLE "
                    f"(got thread level {thread_level})"
                )

        import torch_mpi_ext

        send_tensors = [t for t, _, _ in self._sends]
        dst_ranks = [r for _, r, _ in self._sends]
        send_tags = [tag for _, _, tag in self._sends]
        recv_tensors = [t for t, _, _ in self._recvs]
        src_ranks = [r for _, r, _ in self._recvs]
        recv_tags = [tag for _, _, tag in self._recvs]

        try:
            with device_stream(self._stream):
                torch_mpi_ext.ops.batch_isend_irecv(
                    send_tensors,
                    dst_ranks,
                    send_tags,
                    recv_tensors,
                    src_ranks,
                    recv_tags,
                    self._comm_ptr_wrapper,
                )
        finally:
            self._sends.clear()
            self._recvs.clear()
            self._pair_seq.clear()
            self._used_peer_tags.clear()
