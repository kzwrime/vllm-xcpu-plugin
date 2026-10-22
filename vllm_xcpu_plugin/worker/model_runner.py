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

import vllm_xcpu_plugin.envs as envs_xcpu
from vllm_xcpu_plugin import shared_kv_cache

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
        shared_kv_cache.validate_config(vllm_config)
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
        shared_kv_cache.validate_config(vllm_config)
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self._af_forward_allreduce = envs_xcpu.VLLM_XCPU_AF_FORWARD_ALLREDUCE

    def allocate_kv_cache(self, *args, **kwargs):
        if not shared_kv_cache.enabled():
            return super().allocate_kv_cache(*args, **kwargs)
        draft_layers = getattr(self.speculator, "draft_attn_layer_names", ())
        shared = shared_kv_cache.SharedKVCache(
            self.vllm_config, self.device, draft_layers
        )
        return shared.allocate(*args, **kwargs)

    def update_requests(self, scheduler_output):
        shared = shared_kv_cache.runtime()
        if shared is None or not (
            scheduler_output.new_block_ids_to_zero
            or scheduler_output.kv_cache_block_copies
        ):
            # Forward cache writes store identical bytes on every rank.
            return super().update_requests(scheduler_output)
        # Zeroing and CoW copies store different bytes into recycled blocks. Every
        # rank must finish reading them first and finish rewriting them before any
        # rank appends this step's tokens. TP ranks share the scheduler output, so
        # they all take this branch together.
        shared.fence()
        super().update_requests(scheduler_output)
        shared.fence()

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ):
        if self._af_forward_allreduce and (
            dummy_run or scheduler_output.total_num_scheduled_tokens > 0
        ):
            from vllm_xcpu_plugin.af_ep.attn.runtime import (
                get_remote_experts_client,
            )

            client = get_remote_experts_client()
            if client is not None:
                client.sync_forward_entry()

        return super().execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
            context_len=context_len,
            valid_dummy_state_slots=valid_dummy_state_slots,
            randomize_inputs=randomize_inputs,
        )


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
