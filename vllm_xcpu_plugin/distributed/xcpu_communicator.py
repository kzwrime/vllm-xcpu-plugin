# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.all2all import (
    All2AllManagerBase,
    NaiveAll2AllManager,
)
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

import vllm_xcpu_plugin.envs as envs_xcpu

logger = init_logger(__name__)


class CpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        self.get_cpu_view_from_mcpu_tensor = torch.mcpu.get_cpu_view_from_mcpu_tensor  # type: ignore
        self.get_mcpu_view_from_cpu_tensor = torch.mcpu.get_mcpu_view_from_cpu_tensor  # type: ignore

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
            elif self.all2all_backend == "torch_all_to_all_single":  # type: ignore[has-type]
                self.all2all_manager = All2AllManagerBase(cpu_group=self.cpu_group)
            else:
                raise ValueError(
                    f"Unknown/Unsupported all2all backend: {self.all2all_backend}"
                )
            logger.info("Using all2all_backend = %s", self.all2all_backend)

    def _requires_cpu_staging(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type != "cpu"

    def _to_cpu_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._requires_cpu_staging(tensor):
            if self.get_cpu_view_from_mcpu_tensor is not None:
                return self.get_cpu_view_from_mcpu_tensor(tensor)
            return tensor.to("cpu")
        return tensor

    # def _to_device_from_cpu(
    #     self, tensor: torch.Tensor, device: torch.device
    # ) -> torch.Tensor:
    #     if device.type == "cpu":
    #         return tensor
    #     return tensor.to(device)

    def all_reduce(self, input_):
        if self._requires_cpu_staging(input_):
            cpu_input = self._to_cpu_from_device(input_)
            torch.distributed.all_reduce(cpu_input, group=self.cpu_group)
            return input_
        torch.distributed.all_reduce(input_, group=self.cpu_group)
        return input_

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        cpu_input = self._to_cpu_from_device(input_)
        if self.rank_in_group == dst:
            gather_list = [
                self._to_cpu_from_device(torch.empty_like(input_))
                for _ in range(world_size)
            ]
        else:
            gather_list = None

        torch.distributed.gather(
            cpu_input, gather_list, dst=self.ranks[dst], group=self.cpu_group
        )

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        cpu_input = self._to_cpu_from_device(input_)
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        cpu_output_tensor = self._to_cpu_from_device(output_tensor)
        torch.distributed.all_gather_into_tensor(
            cpu_output_tensor, cpu_input, group=self.cpu_group
        )

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not self._requires_cpu_staging(input_):
            return super().reduce_scatter(input_, dim)

        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        input = input_.movedim(0, dim).contiguous()
        assert input.shape[0] % world_size == 0
        cpu_input = self._to_cpu_from_device(input)
        chunk_size = cpu_input.shape[0] // world_size
        output_shape = (chunk_size,) + cpu_input.shape[1:]
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        cpu_output = self._to_cpu_from_device(output)
        torch.distributed.reduce_scatter_tensor(
            cpu_output, cpu_input, group=self.cpu_group
        )
        return output.movedim(0, dim).contiguous()

    def dispatch(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,  # type: ignore[call-arg]
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(
            hidden_states, is_sequence_parallel
        )
        return hidden_states
