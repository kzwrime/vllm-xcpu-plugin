"""vLLM FP8 MoE integration backed by torch_xcpu."""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _prepack_expert_weight(weight: torch.Tensor, backend: str) -> torch.Tensor:
    packed = torch.empty_like(weight)
    op = getattr(
        torch.ops.torch_xcpu, f"fp8_moe_prepack_weight_{backend}_out"
    ).default
    op(packed, weight.contiguous())
    return packed


def _run_fp8_moe(
    hidden_states: torch.Tensor,
    packed_w13: torch.Tensor,
    packed_w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: list[int],
    backend: str,
) -> torch.Tensor:
    out = torch.empty_like(hidden_states)
    op = getattr(torch.ops.torch_xcpu, f"fp8_moe_{backend}_out").default
    op(
        out,
        hidden_states.contiguous(),
        packed_w13,
        packed_w2,
        w13_scale.contiguous(),
        w2_scale.contiguous(),
        topk_weights.float().contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        block_shape,
    )
    return out


class XcpuExpertsFp8(mk.FusedMoEExpertsMonolithic):
    """XCPU FP8 W8A16 block-quantized monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        backend: str,
        scale_block_size: list[int],
    ):
        super().__init__(moe_config, quant_config)
        self.backend = backend
        if len(scale_block_size) != 2 or min(scale_block_size) <= 0:
            raise ValueError(
                "XCPU FP8 MoE scale block size must contain two positive values"
            )
        self.scale_block_size = tuple(scale_block_size)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        import torch_xcpu

        return (
            current_platform.device_name == "mcpu"
            and torch_xcpu.ops.fp8_moe_supported()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
            and not moe_parallel_config.is_sequence_parallel
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in (
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
            select_experts,
        )

        if activation != MoEActivation.SILU:
            raise ValueError("XCPU FP8 MoE only supports SiLU")
        if expert_map is not None:
            raise NotImplementedError(
                "XCPU FP8 MoE does not yet support expert parallelism"
            )
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "XCPU FP8 MoE applies router weights on output"
            )
        if a1q_scale is not None:
            raise ValueError("XCPU FP8 W8A16 expects unquantized activations")

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor
                if routed_scaling_factor is not None
                else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("XCPU FP8 MoE requires weight scales")
        return _run_fp8_moe(
            hidden_states,
            w1,
            w2,
            self.w1_scale,
            self.w2_scale,
            topk_weights,
            topk_ids,
            list(self.scale_block_size),
            self.backend,
        )


class XcpuFp8MoEMethod(Fp8MoEMethod):
    """FP8 checkpoint loader and monolithic MoE kernel for XCPU."""

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        mk_config = layer.moe_config
        FusedMoEMethodBase.__init__(self, mk_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        if not quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("XCPU FP8 MoE requires a serialized FP8 checkpoint")
        if self.weight_block_size != [128, 128]:
            raise ValueError(
                "XCPU FP8 MoE currently requires 128x128 block-scaled weights"
            )
        if quant_config.activation_scheme != "dynamic":
            raise ValueError("XCPU FP8 MoE requires dynamic activations")
        if mk_config.has_bias:
            raise NotImplementedError("XCPU FP8 MoE does not support expert bias")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "XCPU FP8 MoE does not support router weights on input"
            )

        supported, reason = XcpuExpertsFp8.is_supported_config(
            XcpuExpertsFp8,
            mk_config,
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            mk.FusedMoEActivationFormat.Standard,
        )
        if not supported:
            raise NotImplementedError(f"XCPU FP8 MoE is unsupported: {reason}")

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig:
        return fp8_w8a16_moe_quant_config(
            w1_scale=getattr(layer, f"w13_{self.weight_scale_name}"),
            w2_scale=getattr(layer, f"w2_{self.weight_scale_name}"),
            block_shape=self.weight_block_size,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        import torch_xcpu

        if w13_input_scale is not None or w2_input_scale is not None:
            raise ValueError("XCPU FP8 W8A16 does not use input scales")

        backend = torch_xcpu.ops.resolve_fp8_moe_backend()
        packed_w13 = _prepack_expert_weight(w13, backend)
        packed_w2 = _prepack_expert_weight(w2, backend)
        replace_parameter(layer, "w13_weight", packed_w13)
        replace_parameter(layer, "w2_weight", packed_w2)
        replace_parameter(
            layer,
            f"w13_{self.weight_scale_name}",
            w13_scale.contiguous(),
        )
        replace_parameter(
            layer,
            f"w2_{self.weight_scale_name}",
            w2_scale.contiguous(),
        )

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
            use_monolithic=True,
        )
        assert prepare_finalize is not None
        if (
            prepare_finalize.activation_format
            != mk.FusedMoEActivationFormat.Standard
        ):
            raise NotImplementedError(
                "XCPU FP8 MoE only supports standard activation format"
            )
        assert self.weight_block_size is not None
        self.moe_kernel = mk.FusedMoEKernel(
            prepare_finalize,
            XcpuExpertsFp8(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                backend=backend,
                scale_block_size=self.weight_block_size,
            ),
        )


class XcpuFp8Config(Fp8Config):
    """Use XCPU's FP8 MoE method while preserving vLLM's other FP8 methods."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        method = super().get_quant_method(layer, prefix)
        if (
            current_platform.device_name == "mcpu"
            and isinstance(layer, RoutedExperts)
            and isinstance(method, Fp8MoEMethod)
        ):
            return XcpuFp8MoEMethod(self, layer)
        return method


_quantization_registered = False


def register_fp8_moe_quantization() -> None:
    """Register the platform-scoped FP8 config through vLLM's public API."""
    global _quantization_registered
    if _quantization_registered:
        return
    register_quantization_config("fp8")(XcpuFp8Config)
    _quantization_registered = True
