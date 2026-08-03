# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_xcpu_plugin.envs as envs_xcpu

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

faulthandler.enable()
logger = init_logger(__name__)


class McpuPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "mcpu"
    device_type: str = "privateuseone"
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "cpu:gloo,mcpu:mcpu"
    simple_compile_backend: str = "eager"

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        import vllm_xcpu_plugin.layers.fused_moe.prepare_finalize_factory  # noqa: F401
        from vllm_xcpu_plugin.layers.fp8_moe import (
            register_fp8_moe_quantization,
        )

        register_fp8_moe_quantization()

    @classmethod
    def import_ir_kernels(cls) -> None:
        import vllm_xcpu_plugin.ir_kernels  # noqa: F401

    @classmethod
    def get_default_ir_op_priority(
        cls, vllm_config: "VllmConfig"
    ) -> "IrOpPriorityConfig":
        from vllm.config import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        # Eager execution dispatches directly to the implementation, so it can
        # safely reuse the donated activation buffers. Mcpu compile currently
        # uses DYNAMO_TRACE_ONCE rather than the custom VLLM_COMPILE backend;
        # consequently the vLLM IR functionalization/lowering passes do not
        # run. Select the genuinely functional one-launch kernel before Dynamo
        # tracing instead of exposing a mutating implementation to Inductor.
        fused_priority = ["torch_xcpu", "native"]
        if vllm_config.compilation_config.mode in (
            None,
            CompilationMode.NONE,
        ):
            fused_priority.insert(0, "torch_xcpu_inplace")

        return IrOpPriorityConfig.with_default(
            ["torch_xcpu", "native"],
            fused_add_rms_norm=fused_priority,
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:

        if selected_backend:
            # logger.info("Cannot use %s backend on CPU.", selected_backend)
            logger.info("Using %s backend on MCPU", selected_backend)
            if attn_selector_config.use_mla:
                assert selected_backend == AttentionBackendEnum.TRITON_MLA, (
                    f"MLA is enabled, but selected backend is {selected_backend}."
                )
            return selected_backend.get_path()

        if attn_selector_config.use_mla:
            return AttentionBackendEnum.TRITON_MLA.get_path()
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on CPU.")

        return AttentionBackendEnum.TRITON_ATTN.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.TRITON_ATTN,
        ]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        cast(Any, torch).mcpu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        cast(Any, torch).mcpu.manual_seed_all(seed)

    @classmethod
    def current_device(cls) -> torch.device:
        """
        Return the torch device used for tensors allocated by vLLM.

        vLLM model code calls current_platform.current_device() when creating
        some modules.  The xcpu backend is registered as PrivateUse1, while the
        user-facing name is mcpu, so use the canonical torch device type here.
        """
        return torch.device(cls.device_type)

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        # capacity format differs from cuda's and will cause unexpected
        # failure, so use None directly
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"mcpu:{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        from vllm.utils.mem_constants import GiB_bytes

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        assert kv_cache_space is not None, (
            "VLLM_CPU_KVCACHE_SPACE must be set for MCPU backend."
        )
        kv_cache_space *= GiB_bytes
        return kv_cache_space

        # device_props = torch.mcpu.get_device_properties(device_id)  # type: ignore
        # return device_props.total_memory

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    # @classmethod
    # def get_static_graph_wrapper_cls(cls) -> str:
    #     return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        vllm_config.compilation_config.custom_ops = ["all"]
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "vllm_xcpu_plugin.worker.worker_v1.McpuWorker"

        cache_config = vllm_config.cache_config
        if not cache_config.user_specified_block_size:
            model_config = vllm_config.model_config
            is_hybrid_model = model_config is not None and model_config.is_hybrid
            # Hybrid models (Qwen3.5 in this repo) rely on
            # HybridAttentionMambaModelConfig to align block_size with
            # mamba/attention page sizes. Keep the model-side value.
            if not is_hybrid_model:
                cache_config.block_size = 64

        # Align block_size to envs.BLOCK_SIZE_ALIGN
        block_size_align = envs_xcpu.BLOCK_SIZE_ALIGN
        cache_config.block_size = (
            (cache_config.block_size + block_size_align - 1) // block_size_align
        ) * block_size_align
        # Note: workaround for v1 gpu_model_runner
        from vllm.config import CompilationMode

        compilation_config = vllm_config.compilation_config
        if vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
            compilation_config.backend = "inductor"
            compilation_config.inductor_compile_config.update({
                "dce": True,
                "size_asserts": False,
                "nan_asserts": False,
                # mcpu memory must only be touched by stream-aware ATen/custom
                # operators. Do not let Inductor synthesize host-side fused
                # compute kernels; metadata-only views are still lowered.
                "epilogue_fusion": False,
                "pattern_matcher": False,
                "cpp.dynamic_threads": True,
                # Inductor combo kernels currently require a SIMD/CUDA
                # scheduler. mcpu delegates to the C++ CPU scheduler, so keep
                # this off unless the mcpu backend grows combo-kernel codegen.
                "combo_kernels": False,
                "benchmark_combo_kernel": False,
            })

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # TODO: CPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        # CUDA, ROCM, XPU return True
        return False

    @classmethod
    def is_cpu(cls) -> bool:
        # mcpu is CPU-emulated hardware; return True so that vLLM treats it
        # like a CPU backend and skips combo_kernels (which require SIMDScheduling).
        return False

        # return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # TODO return True
        return True
        # return False

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # This method is used by vLLM's DeviceMemoryProfiler around model
        # loading. On mcpu, empty_cache() is a destructive allocator operation:
        # it synchronizes and releases cached protected pages, which can block
        # worker startup after the model weights have been loaded. We only need
        # allocator accounting here, so avoid releasing the cache while sampling.
        torch.accelerator.synchronize(device)
        torch.accelerator.reset_peak_memory_stats(device)
        return torch.accelerator.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if envs_xcpu.VLLM_CPU_USE_MPI:
            return (
                "vllm_xcpu_plugin.distributed.cpu_mpi_communicator.CpuMPICommunicator"  # noqa
            )
        return "vllm_xcpu_plugin.distributed.xcpu_communicator.CpuCommunicator"  # noqa

    @classmethod
    def device_count(cls) -> int:
        return torch.accelerator.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype not in [torch.float, torch.bfloat16]:  # noqa: SIM102
            raise ValueError(
                f"mcpu only support float32 and bfloat16, but got {dtype}."
            )

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return False

    # @classmethod
    # def insert_blocks_to_device(
    #     cls,
    #     src_cache: torch.Tensor,
    #     dst_cache: torch.Tensor,
    #     src_block_indices: torch.Tensor,
    #     dst_block_indices: torch.Tensor,
    # ) -> None:
    #     """Copy blocks from src_cache to dst_cache on XPU."""
    #     _src_cache = src_cache[:, src_block_indices]
    #     dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    # @classmethod
    # def swap_out_blocks_to_host(
    #     cls,
    #     src_cache: torch.Tensor,
    #     dst_cache: torch.Tensor,
    #     src_block_indices: torch.Tensor,
    #     dst_block_indices: torch.Tensor,
    # ) -> None:
    #     """Copy blocks from XPU to host (CPU)."""
    #     _src_cache = src_cache[:, src_block_indices]
    #     dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return 4

    @classmethod
    def memory_stats(cls, device_index=None, /) -> OrderedDict[str, Any]:
        return torch.accelerator.memory_stats(device_index)

    @classmethod
    def mem_get_info(cls, device_index=None, /) -> tuple[int, int]:
        return torch.accelerator.get_memory_info(device_index)
