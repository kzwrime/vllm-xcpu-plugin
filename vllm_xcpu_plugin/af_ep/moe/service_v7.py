"""F-rank execution endpoint with one service-owned AF V7 workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm_xcpu_plugin.distributed.mpi_world import ClusterType

from ..common.session_v7 import AfV7Session

if TYPE_CHECKING:
    from vllm_xcpu_plugin.layers.fused_moe.routed_experts import XcpuRoutedExperts

    from .model import RoutedExpertsModel


@dataclass(frozen=True)
class _ExpertLayer:
    layer_idx: int
    backend: Any
    expert_map: torch.Tensor


class ExpertServiceV7:
    """Serialize receive -> compute -> send on one stream, reusing all scratch.

    MPI receive storage remains session-owned. The native transaction owns its
    internal workspace views; Python only binds an operator and allocates bytes.
    """

    def __init__(
        self,
        model: RoutedExpertsModel,
        session: AfV7Session,
        *,
        compile_model: bool = False,
    ) -> None:
        assert session.cluster_type == ClusterType.MOE
        assert session.ep_size == model.ep_size
        assert session.role_rank == model.ep_rank
        self.model = model
        self.session = session
        self._layers = tuple(self._prepare_layer(i) for i in model.layer_indices)
        backend_type = self._layers[0].backend.backend_type
        if any(
            layer.backend.backend_type is not backend_type for layer in self._layers
        ):
            raise ValueError("AF service requires one compute backend across layers")
        self._op = getattr(
            torch.ops.torch_xcpu, f"fused_af_f_moe_v7_{backend_type.__name__}"
        ).default
        self._size_op = getattr(
            torch.ops.torch_xcpu,
            f"fused_af_f_moe_v7_workspace_size_{backend_type.__name__}",
        )
        self._workspace: torch.Tensor | None = None
        self._run_layer = self._execute_layer
        if compile_model:
            self._run_layer = torch.compile(
                self._execute_layer,
                backend="inductor",
                fullgraph=True,
                dynamic=True,
                options={
                    "epilogue_fusion": False,
                    "pattern_matcher": False,
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            )

    def initialize(self) -> None:
        # Negotiate the A-side post-SP capacity before allocating F storage.
        self.session.initialize(
            self.model.hidden_size, self.model.top_k, self.model.dtype
        )
        if self._workspace is not None:
            return
        sizes = []
        for layer in self._layers:
            params = layer.backend.params
            gemm1, gemm2 = params.gemm1.params, params.gemm2.params
            block_n, block_k = gemm1.scale_block_size or (0, 0)
            sizes.append(
                self._size_op(
                    self.session.num_attention_ranks,
                    self.session.max_rows_per_attention_rank,
                    self.model.top_k,
                    self.model.hidden_size,
                    self.model.intermediate_size,
                    self.model.num_experts // self.model.ep_size,
                    self.model.dtype.itemsize,
                    block_n,
                    block_k,
                    gemm1.bias is not None,
                    gemm2.bias is not None,
                )
            )
        self._workspace = torch.empty(
            max(sizes), dtype=torch.uint8, device=self.model.device
        )

        print(
            f"AF-EP V7 F{self.session.role_rank} "
            f"workspace_bytes={self._workspace.numel()} "
            f"input_capacity={self.session.expert_capacity}",
            flush=True,
        )

    def execute_model_pass(self) -> None:
        self.session.validate_initialized(
            self.model.hidden_size, self.model.top_k, self.model.dtype
        )
        self.session.sync_forward_entry()
        for layer in self._layers:
            self._run_layer(layer)
        # Keep the infinite service loop from accumulating unbounded submissions.
        torch.accelerator.synchronize()

    def _execute_layer(self, layer: _ExpertLayer) -> None:
        assert self._workspace is not None
        params = layer.backend.params
        gemm1, gemm2 = params.gemm1.params, params.gemm2.params
        block_n, block_k = gemm1.scale_block_size or (0, 0)
        self._op(
            gemm1.packed_weight,
            gemm2.packed_weight,
            gemm1.packed_weight_scale,
            gemm2.packed_weight_scale,
            gemm1.bias,
            gemm2.bias,
            block_n,
            block_k,
            self.model.num_experts,
            layer.expert_map,
            self.session.max_rows_per_attention_rank,
            self.model.top_k,
            self.model.dtype.itemsize,
            layer.layer_idx,
            self.session.metadata,
            self.session.communicator_handle,
            self._workspace,
        )

    def _prepare_layer(self, layer_idx: int) -> _ExpertLayer:
        layer = cast("XcpuRoutedExperts", self.model.routed_experts[str(layer_idx)])
        if layer.expert_map is None:
            raise ValueError("AF experts require a global-to-local expert map")
        expert_map = layer.expert_map.to(
            device=self.model.device, dtype=torch.int32
        ).contiguous()
        backend = self.model.fused_moe_for_layer(layer_idx)
        params = backend.params
        if (
            params.hidden != self.model.hidden_size
            or params.intermediate != self.model.intermediate_size
            or params.experts != self.model.num_experts // self.model.ep_size
        ):
            raise ValueError("AF compute dimensions disagree with the service")
        if params.gemm1.params.scale_block_size != params.gemm2.params.scale_block_size:
            raise ValueError("AF GEMMs require matching scale block sizes")
        if self.model.hidden_act not in ("silu", "swiglu"):
            raise ValueError("AF experts require SiLU-and-mul")
        return _ExpertLayer(layer_idx, backend, expert_map)
