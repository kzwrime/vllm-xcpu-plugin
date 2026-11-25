# SPDX-License-Identifier: Apache-2.0
"""XCPU W8A16 quantization config and methods for Int8 + UE8M0 block scales."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    validate_fp8_block_shape,
    validate_fp8_block_shape_moe,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from .fused_moe.grouped_gemm_experts import XcpuGroupedGemmExperts
from .fused_moe.setup import (
    initialize_padded_expert_slots,
    install_fused_moe,
    reject_fused_moe_hot_reload,
    use_shared_moe_workspace,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_DEFAULT_BLOCK_SIZE = [16, 16]


def _initialize_dummy_ue8m0_scale(scale: torch.Tensor) -> None:
    config = get_current_vllm_config_or_none()
    if config is not None and config.load_config.load_format == "dummy":
        # DummyModelLoader leaves uint8 tensors uninitialized; 127 encodes 2^0.
        scale.data.view(torch.uint8).fill_(127)


def _dequantize_int8_ue8m0_cpu(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    scales = weight_scale.detach().cpu().view(torch.float8_e8m0fnu).to(torch.float32)
    expanded = scales.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)[
        : weight.shape[0], : weight.shape[1]
    ]
    dequant = weight.detach().cpu().to(torch.float32) * expanded
    return dequant.to(device=weight.device, dtype=out_dtype)


class XcpuInt8Ue8m0Config(QuantizationConfig):
    """Quantization config for W8A16 signed int8 weights + UE8M0 block scales."""

    def __init__(
        self,
        fmt: str = "int8_ue8m0",
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
    ) -> None:
        super().__init__()
        if fmt != "int8_ue8m0":
            raise ValueError(f"Unsupported xcpu_int8 fmt: {fmt!r}")
        if activation_scheme != "dynamic":
            raise ValueError(
                "xcpu_int8 only supports dynamic/unquantized activations "
                f"(W8A16), got {activation_scheme!r}"
            )
        block_size = (
            list(_DEFAULT_BLOCK_SIZE)
            if weight_block_size is None
            else [int(v) for v in weight_block_size]
        )
        if len(block_size) != 2 or min(block_size) <= 0:
            raise ValueError(
                "xcpu_int8 weight_block_size must contain 2 positive ints, "
                f"got {block_size}"
            )
        self.fmt = fmt
        self.activation_scheme = activation_scheme
        self.ignored_layers = list(ignored_layers or [])
        self.weight_block_size = block_size

    def get_name(self) -> Any:
        return "xcpu_int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> XcpuInt8Ue8m0Config:
        fmt = cls.get_from_keys_or(config, ["fmt"], "int8_ue8m0")
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        weight_block_size = cls.get_from_keys_or(
            config, ["weight_block_size"], _DEFAULT_BLOCK_SIZE
        )
        return cls(
            fmt=fmt,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return XcpuInt8Ue8m0LinearMethod(self)
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return XcpuInt8Ue8m0MoEMethod(self, layer)
        return None


class XcpuInt8Ue8m0LinearMethod(LinearMethodBase):
    """W8A16 Int8 + UE8M0 linear method backed by ``torch_xcpu``."""

    supports_pre_processed_weights = False

    def __init__(self, quant_config: XcpuInt8Ue8m0Config) -> None:
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer_any: Any = layer
        layer_any.logical_widths = output_partition_sizes
        layer_any.input_size_per_partition = input_size_per_partition
        layer_any.output_size_per_partition = output_size_per_partition
        layer_any.orig_dtype = params_dtype
        layer_any.weight_block_size = self.weight_block_size

        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        block_n, block_k = self.weight_block_size
        scale = BlockQuantScaleParameter(
            data=torch.full(
                (
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                ),
                127,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        set_weight_attrs(scale, {"scale_type": "weight_scale"})
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        layer_any: Any = layer
        weight_scale = layer_any.weight_scale_inv.data.contiguous()
        _initialize_dummy_ue8m0_scale(weight_scale)

        bn, bk = self.weight_block_size
        if getattr(layer_any, "_xcpu_mla_weight_only", False):
            w = layer_any.weight.data
            if (
                w.device.type == "privateuseone"
                and layer_any.params_dtype == torch.bfloat16
                and w.shape[0] % bn == 0
                and w.shape[1] % bk == 0
                and bk % 16 == 0
            ):
                dequant = torch_xcpu.ops.int8_block_dequant(
                    w.contiguous(), weight_scale, block_n=bn, block_k=bk
                )
            else:
                dequant = _dequantize_int8_ue8m0_cpu(
                    w, weight_scale, bn, bk, layer_any.params_dtype
                )
            replace_parameter(layer, "weight", dequant)
            return

        layer_bias = getattr(layer, "bias", None)
        if getattr(layer, "skip_bias_add", False) or (
            isinstance(layer, RowParallelLinear) and layer.tp_rank > 0
        ):
            layer_bias = None
        bias_fp32 = None
        if layer_bias is not None:
            assert isinstance(layer_bias, torch.Tensor)
            bias_fp32 = layer_bias.float().contiguous()

        int8_linear = torch_xcpu.ops.initialize_int8_linear(
            layer.weight,
            weight_scale,
            (bn, bk),
            bias_fp32,
        )
        logger.warning_once(
            "Using torch_xcpu INT8 UE8M0 linear backend=%s",
            int8_linear.params.backend.name.lower(),
            scope="process",
        )

        replace_parameter(layer, "weight", int8_linear.params.packed_weight)
        replace_parameter(layer, "weight_scale_inv", int8_linear.params.weight_scale)
        if bias_fp32 is not None:
            replace_parameter(layer, "bias", bias_fp32)
        layer._xcpu_int8_linear = int8_linear

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        int8_linear = getattr(layer, "_xcpu_int8_linear", None)
        if int8_linear is None:
            raise RuntimeError(
                "torch_xcpu INT8 linear was not initialized; "
                "process_weights_after_loading() must run before inference"
            )
        if (bias is not None) != (int8_linear.params.bias is not None):
            raise RuntimeError(
                "The runtime bias does not match the bias used to initialize "
                "torch_xcpu INT8 linear"
            )
        return int8_linear(x)


class XcpuInt8Ue8m0MoEMethod(FusedMoEMethodBase):
    """W8A16 Int8 + UE8M0 MoE method for the unified XCPU MoE pipeline."""

    supports_pre_processed_weights = False

    def __init__(self, quant_config: XcpuInt8Ue8m0Config, layer: RoutedExperts) -> None:
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        if layer.moe_config.has_bias:
            raise NotImplementedError("XCPU INT8 MoE expert bias is unsupported")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "XCPU INT8 MoE applies router weights after expert compute"
            )
        block_n, block_k = self.weight_block_size
        self.weight_quant_key = QuantKey(
            dtype=torch.int8,
            scale=ScaleDesc(
                dtype=torch.uint8,
                static=True,
                group_shape=GroupShape(block_n, block_k),
            ),
            symmetric=True,
        )
        supported, reason = XcpuGroupedGemmExperts.is_supported_config(
            XcpuGroupedGemmExperts,
            layer.moe_config,
            self.weight_quant_key,
            None,
            FusedMoEActivationFormat.Standard,
        )
        if not supported:
            raise NotImplementedError(f"XCPU INT8 MoE is unsupported: {reason}")

    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        validate_fp8_block_shape_moe(
            intermediate_size_per_partition,
            self.weight_block_size,
            tp_size=self.moe.tp_size,
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        block_n, block_k = self.weight_block_size
        w13_weight_scale_inv = torch.nn.Parameter(
            torch.full(
                (
                    num_experts,
                    self.moe.w13_num_shards
                    * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                ),
                127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        w2_weight_scale_inv = torch.nn.Parameter(
            torch.full(
                (
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                ),
                127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale_inv)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale_inv)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        set_weight_attrs(w13_weight_scale_inv, scale_attrs)
        set_weight_attrs(w2_weight_scale_inv, scale_attrs)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        bn, bk = self.weight_block_size
        group_shape = GroupShape(bn, bk)
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(torch.int8, group_shape, layer.w13_weight_scale_inv),
            _w2=FusedMoEQuantDesc(torch.int8, group_shape, layer.w2_weight_scale_inv),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import torch_xcpu

        if not isinstance(layer, RoutedExperts):
            raise TypeError("XCPU INT8 MoE method requires RoutedExperts")
        reject_fused_moe_hot_reload(self)
        initialize_padded_expert_slots(layer)
        w13_scale = layer.w13_weight_scale_inv
        w2_scale = layer.w2_weight_scale_inv
        assert isinstance(w13_scale, torch.Tensor)
        assert isinstance(w2_scale, torch.Tensor)
        _initialize_dummy_ue8m0_scale(w13_scale)
        _initialize_dummy_ue8m0_scale(w2_scale)

        fused_moe = torch_xcpu.ops.initialize_fused_moe(
            layer.w13_weight,
            layer.w2_weight,
            w13_scale,
            w2_scale,
            scale_block_size=self.weight_block_size,
            m_capacity=self.moe.max_num_tokens * self.moe.experts_per_token,
            allocate_scratch=not use_shared_moe_workspace(self.moe),
        )
        install_fused_moe(
            self,
            layer,
            fused_moe,
            self.get_fused_moe_quant_config,
            scale_names=("w13_weight_scale_inv", "w2_weight_scale_inv"),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


def register_xcpu_int8_quantization() -> None:
    """Register ``xcpu_int8`` quantization config once idempotently."""
    if "xcpu_int8" not in QUANTIZATION_METHODS:
        register_quantization_config("xcpu_int8")(XcpuInt8Ue8m0Config)
