"""Platform-scoped RoutedExperts replacement for XCPU MoE methods."""

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.platforms import current_platform


def xcpu_moe_method_factory(
    upstream: FusedMoEMethodBase,
    quant_config: QuantizationConfig | None,
    layer: RoutedExperts,
) -> FusedMoEMethodBase:
    """Map only the supported upstream MoE method families."""
    if isinstance(upstream, Fp8MoEMethod):
        if not isinstance(quant_config, Fp8Config):
            raise TypeError("upstream FP8 MoE method requires Fp8Config")
        from vllm_xcpu_plugin.layers.fp8_moe import XcpuFp8MoEMethod

        return XcpuFp8MoEMethod(quant_config, layer)
    if isinstance(upstream, CompressedTensorsW4A4Mxfp4MoEMethod):
        from vllm_xcpu_plugin.layers.mxfp4_moe import (
            XcpuCompressedTensorsMxfp4MoEMethod,
        )

        return XcpuCompressedTensorsMxfp4MoEMethod(layer.moe_config)
    if type(upstream) is Mxfp4MoEMethod:
        from vllm_xcpu_plugin.layers.mxfp4_moe import XcpuMxfp4MoEMethod

        return XcpuMxfp4MoEMethod(layer.moe_config)
    if isinstance(upstream, QuarkOCP_MX_MoEMethod):
        from vllm_xcpu_plugin.layers.quark_mxfp4 import (
            XcpuQuarkOCPMXMoEMethod,
            should_use_quark_mxfp4_w4a16,
        )

        if not should_use_quark_mxfp4_w4a16(
            upstream.weight_dtype, upstream.input_dtype
        ):
            return upstream

        return XcpuQuarkOCPMXMoEMethod(
            layer.moe_config,
            upstream.weight_quant_key,
            None,
        )
    if isinstance(upstream, UnquantizedFusedMoEMethod):
        from .unquantized_fused_moe_method import (
            XcpuUnquantizedFusedMoEMethod,
        )

        return XcpuUnquantizedFusedMoEMethod(layer.moe_config)
    return upstream


def _apply_padded_expert_map(
    expert_map_manager,
    padded_global: int,
    padded_local: int,
    start: int,
) -> None:
    import torch

    existing_map = expert_map_manager._expert_map
    map_device = existing_map.device if existing_map is not None else "cpu"
    map_dtype = existing_map.dtype if existing_map is not None else torch.int32
    padded_map = torch.full((padded_global,), -1, dtype=map_dtype, device=map_device)
    padded_map[start : start + padded_local] = torch.arange(
        padded_local, dtype=map_dtype, device=map_device
    )
    expert_map_manager.global_num_experts = padded_global
    expert_map_manager._local_num_experts = padded_local
    expert_map_manager._expert_map = padded_map


def _maybe_pad_ep_experts(
    moe_config: FusedMoEConfig,
    expert_map_manager,
) -> tuple[bool, int | None, int]:
    """Uniformly pad non-divisible EP expert counts for XCPU MPI V5/V6.

    FusedMoE V5/V6 requires equal local_num_experts across ranks. When
    unpadded num_experts is not divisible by ep_size, pad up to
    ceil(unpadded / ep_size) * ep_size using linear contiguous ownership and
    zeroed/unit-scaled trailing expert slots.
    """
    import torch
    from vllm.logger import init_logger

    existing = getattr(moe_config, "_xcpu_ep_padding_info", None)
    if existing is not None:
        unpadded_global, padded_global, padded_local = existing
        start = moe_config.ep_rank * padded_local
        end = min(unpadded_global, start + padded_local)
        active_local = max(0, end - start)
        if expert_map_manager.global_num_experts != padded_global:
            _apply_padded_expert_map(
                expert_map_manager, padded_global, padded_local, start
            )
        return True, active_local, unpadded_global

    unpadded_global = moe_config.num_experts
    parallel = moe_config.moe_parallel_config
    ep_size = moe_config.ep_size
    if (
        torch.device(moe_config.device).type != "mcpu"
        or not moe_config.use_ep
        or ep_size <= 1
        or unpadded_global % ep_size == 0
        or parallel.all2all_backend not in {"mpi_alltoallv_v5", "mpi_alltoallv_v6"}
        or parallel.enable_eplb
        or expert_map_manager.placement_strategy != "linear"
        or expert_map_manager.routing_tables is not None
        or expert_map_manager.num_fused_shared_experts != 0
    ):
        return False, None, unpadded_global

    padded_local = (unpadded_global + ep_size - 1) // ep_size
    padded_global = padded_local * ep_size
    start = moe_config.ep_rank * padded_local
    end = min(unpadded_global, start + padded_local)
    active_local = max(0, end - start)

    _apply_padded_expert_map(
        expert_map_manager, padded_global, padded_local, start
    )
    moe_config.num_experts = padded_global
    moe_config.num_local_experts = padded_local
    moe_config._xcpu_ep_padding_info = (
        unpadded_global,
        padded_global,
        padded_local,
    )
    init_logger(__name__).info_once(
        "XCPU EP padded non-divisible experts from %d to %d "
        "(ep_size=%d, local_experts=%d, active_local=%d)",
        unpadded_global,
        padded_global,
        ep_size,
        padded_local,
        active_local,
    )
    return True, active_local, unpadded_global


@RoutedExperts.register_oot
class XcpuRoutedExperts(RoutedExperts):
    """Delegate checkpoint interpretation upstream, then replace MoE compute."""

    def __init__(
        self,
        layer_name: str,
        params_dtype,
        moe_config: FusedMoEConfig,
        quant_config,
        expert_map_manager,
        *args,
        **kwargs,
    ) -> None:
        padded, active_local, unpadded_global = _maybe_pad_ep_experts(
            moe_config, expert_map_manager
        )
        super().__init__(
            layer_name,
            params_dtype,
            moe_config,
            quant_config,
            expert_map_manager,
            *args,
            **kwargs,
        )
        self._xcpu_padded_experts = padded
        self._xcpu_active_local_experts = active_local
        self._xcpu_unpadded_global_experts = unpadded_global

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        from vllm_xcpu_plugin.af_ep.attn.runtime import get_remote_experts_client

        af_client = get_remote_experts_client()
        if af_client is not None:
            from vllm_xcpu_plugin.af_ep.attn.remote_fused_moe import (
                RemoteExpertsFusedMoEMethod,
            )

            return RemoteExpertsFusedMoEMethod(
                moe_config,
                layer_name=prefix,
                client=af_client,
            )
        upstream = super()._get_quant_method(prefix, quant_config, moe_config)
        if current_platform.device_name != "mcpu":
            return upstream
        return xcpu_moe_method_factory(upstream, quant_config, self)

    def get_expert_mapping(
        self,
        ckpt_gate_proj_name: str | None = None,
        ckpt_down_proj_name: str | None = None,
        ckpt_up_proj_name: str | None = None,
        include_fused: bool = False,
    ) -> list[tuple[str, str, int, str]]:
        if not getattr(self, "_xcpu_padded_experts", False):
            return super().get_expert_mapping(
                ckpt_gate_proj_name=ckpt_gate_proj_name,
                ckpt_down_proj_name=ckpt_down_proj_name,
                ckpt_up_proj_name=ckpt_up_proj_name,
                include_fused=include_fused,
            )
        padded_global = self.moe_config.num_experts
        self.moe_config.num_experts = self._xcpu_unpadded_global_experts
        try:
            # Restore logical expert count temporarily so padded slots are not
            # misclassified as EPLB redundant replicas.
            return super().get_expert_mapping(
                ckpt_gate_proj_name=ckpt_gate_proj_name,
                ckpt_down_proj_name=ckpt_down_proj_name,
                ckpt_up_proj_name=ckpt_up_proj_name,
                include_fused=include_fused,
            )
        finally:
            self.moe_config.num_experts = padded_global

    def load_weights(self, weights):
        from vllm_xcpu_plugin.af_ep.attn.remote_fused_moe import (
            RemoteExpertsFusedMoEMethod,
        )

        if isinstance(self.quant_method, RemoteExpertsFusedMoEMethod):
            return self.quant_method.skip_checkpoint_weights(weights)
        loaded = super().load_weights(weights)
        if getattr(self, "_xcpu_active_local_experts", None) == 0:
            return set(loaded) | {
                name for name, _ in self.named_parameters(recurse=False)
            }
        return loaded
