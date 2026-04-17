# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class McpuModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False


class McpuModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    try:
        # Pre-warm TorchDynamo's handler dict BEFORE patching torch.cuda APIs.
        # Patching torch.cuda.current_stream = torch.accelerator.current_stream
        # makes them the same object, which trips an assertion inside
        # TorchInGraphFunctionVariable._get_handlers() (@functools.cache) when
        # it tries to register both as distinct keys in the handlers dict.
        # Calling _get_handlers() here populates the cache with the original
        # distinct function objects so the assertion never fires during compile.
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        TorchInGraphFunctionVariable._get_handlers()

        # replace cuda APIs with mcpu APIs, this should work by default
        torch.cuda.Stream = torch.mcpu.Stream  # type: ignore
        torch.cuda.default_stream = torch.accelerator.current_stream  # type: ignore
        torch.cuda.current_stream = torch.accelerator.current_stream  # type: ignore
        torch.cuda.stream = torch.mcpu.stream  # type: ignore
        torch.cuda.mem_get_info = torch.accelerator.get_memory_info  # type: ignore
        torch.cuda.Event = torch.Event  # type: ignore
        torch.cuda.set_stream = torch.accelerator.set_stream  # type: ignore
        # if supports_xpu_graph():
        #     torch.cuda.graph = torch.xpu.graph
        #     torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        #     torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle
        yield
    finally:
        pass
