"""Platform-scoped RoutedExperts replacement for XCPU MoE methods."""

from collections.abc import Iterable

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
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
    """Uniformly pad non-divisible EP expert counts for XCPU MPI V5/V6/V8.

    FusedMoE V5/V6/V8 requires equal local_num_experts across expert ranks.
    When unpadded num_experts is not divisible by ep_size, pad up to
    ceil(unpadded / ep_size) * ep_size using linear contiguous ownership and
    zeroed/unit-scaled trailing expert slots.
    """
    import torch
    from vllm.logger import init_logger

    from vllm_xcpu_plugin.af_ep.attn.runtime import get_remote_experts_client

    # AF-EP attention ranks hold no routed weights; moe_config.ep_size here is
    # A's DP*TP world size rather than F's EP size. Leave A unpadded so
    # ExpertsClientV8/AfV8Session pads once against F's expert rank count.
    if get_remote_experts_client() is not None:
        return False, None, moe_config.num_experts

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
        or parallel.all2all_backend
        not in {"mpi_alltoallv_v5", "mpi_alltoallv_v6", "mpi_alltoallv_v8"}
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


def _local_redundant_fused_map(layer: "XcpuRoutedExperts") -> list[tuple[int, int]]:
    """Return local ``(physical_id, logical_id)`` redundant pairs for EPLB."""
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is None or not moe_config.moe_parallel_config.enable_eplb:
        return []
    num_redundant = moe_config.num_experts - moe_config.num_logical_experts
    if num_redundant <= 0:
        return []
    from vllm.distributed.eplb.eplb_state import EplbState

    num_fused_shared = layer.expert_map_manager.num_fused_shared_experts
    physical_to_logical = EplbState.build_initial_global_physical_to_logical_map(
        moe_config.num_logical_experts + num_fused_shared,
        num_redundant,
    )
    num_primary = moe_config.num_logical_experts + num_fused_shared
    return [
        (phys_id, physical_to_logical[phys_id])
        for phys_id in range(num_primary, len(physical_to_logical))
        if layer._map_global_expert_id_to_local_expert_id(phys_id) != -1
    ]


def _load_redundant_fused_slices(
    layer: "XcpuRoutedExperts",
    expert_name: str,
    logical_slices: dict[int, torch.Tensor],
    local_redundant: list[tuple[int, int]],
    expert_mapping: list[tuple[str, str, int, str]],
) -> Iterable[str]:
    """Replicate 3D fused checkpoint slices onto local redundant physical slots.

    Upstream ``RoutedExperts.load_weights`` binds 3D fused tensors strictly by
    logical index ``0..num_logical-1`` and never calls ``weight_loader`` for
    redundant physical slots (>= ``num_logical``).
    """
    qual_name = f"{layer.layer_name}.{expert_name}"
    matched = False
    for param_name, weight_name, chunk_id, shard_id in expert_mapping:
        if weight_name not in qual_name:
            if matched:
                break
            continue
        matched = True
        mapped_weight_name = qual_name.replace(weight_name, param_name)
        resolved_param_name = mapped_weight_name.removeprefix(f"{layer.layer_name}.")
        param = getattr(layer, resolved_param_name, None)
        if param is None:
            continue
        quant_method = getattr(param, "quant_method", None)
        uses_weight_layout = (
            "scale" not in mapped_weight_name
            or quant_method == FusedMoeWeightScaleSupported.BLOCK.value
        )
        transpose = layer.is_fused_checkpoint_transposed and uses_weight_layout
        for phys_id, logical_id in local_redundant:
            slice_2d = layer._orient_fused_weight(logical_slices[logical_id], transpose)
            loaded_expert = (
                slice_2d.chunk(2, dim=0)[chunk_id]
                if shard_id in {"w1", "w3"} and layer.moe_config.is_act_and_mul
                else slice_2d
            )
            success = param.weight_loader(
                param=param,
                loaded_weight=loaded_expert,
                weight_name=mapped_weight_name,
                shard_id=shard_id,
                expert_id=phys_id,
                return_success=True,
            )
            if success:
                yield resolved_param_name


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
        if (
            moe_config.moe_parallel_config.enable_eplb
            and moe_config.num_experts % moe_config.ep_size != 0
        ):
            raise ValueError(
                f"XCPU EPLB requires physical num_experts ({moe_config.num_experts}) "
                f"to be divisible by ep_size ({moe_config.ep_size})"
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

        local_redundant = _local_redundant_fused_map(self)
        if local_redundant:
            needed_logical = sorted({logical_id for _, logical_id in local_redundant})
            captured: list[tuple[str, dict[int, torch.Tensor]]] = []

            def _capture_fused_weights():
                for expert_name, loaded_weight in weights:
                    if loaded_weight.dim() == 3:
                        captured.append(
                            (
                                expert_name,
                                {
                                    lid: loaded_weight[lid].clone()
                                    for lid in needed_logical
                                },
                            )
                        )
                    yield expert_name, loaded_weight

            def _load_with_redundant():
                yield from super(XcpuRoutedExperts, self).load_weights(
                    _capture_fused_weights()
                )
                if captured:
                    expert_mapping = self.get_expert_mapping(include_fused=True)
                    for expert_name, logical_slices in captured:
                        yield from _load_redundant_fused_slices(
                            self,
                            expert_name,
                            logical_slices,
                            local_redundant,
                            expert_mapping,
                        )

            loaded = _load_with_redundant()
        else:
            loaded = super().load_weights(weights)

        if getattr(self, "_xcpu_active_local_experts", None) == 0:
            return set(loaded) | {
                name for name, _ in self.named_parameters(recurse=False)
            }
        return loaded

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(super().get_expert_weights())
        fused_moe = getattr(self, "_xcpu_fused_moe", None)
        storage_by_ptr: dict[int, torch.Tensor] = {}
        if fused_moe is not None:
            for gemm in (
                fused_moe.params.gemm1.params,
                fused_moe.params.gemm2.params,
            ):
                if gemm.packed_weight.numel() > 0:
                    storage_by_ptr[gemm.packed_weight.data_ptr()] = (
                        gemm.packed_weight_storage
                    )
        return [
            storage_by_ptr.get(w.data_ptr(), w)
            .view(torch.uint8)
            .view(self.local_num_experts, -1)
            for w in weights
        ]
