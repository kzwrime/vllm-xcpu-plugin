# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 model entry point on XCPU.

The upstream CUDA model (``vllm.models.deepseek_v41.nvidia``) is reused as is:
its layer graph, engram, weight loading, decoder replay and EP/SP handling are
platform neutral. Only the symbols that name CUDA kernels are rebound, and
only on MCPU, after the upstream contracts (``upstream.py``) are verified.
"""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
import vllm._custom_ops as vllm_ops
import vllm.models.common.ops.sequence_parallel as sequence_parallel
import vllm.models.deepseek_v41.attention as v41_attention
import vllm.models.deepseek_v41.compressor as v41_compressor
import vllm.models.deepseek_v41.nvidia.dspark as nv_dspark
import vllm.models.deepseek_v41.nvidia.model as nv_model
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.utils import xcpu_raw_linear_weight
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
)
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM
from vllm.platforms import current_platform

from vllm_xcpu_plugin.engram_disagg.attn.runtime import get_engram_client
from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility
from vllm_xcpu_plugin.upstream_contracts import rebind

from . import ops
from .attention import XcpuDeepseekV41Attention
from .backends import XcpuDeepseekV41IndexerCache
from .sp_aux import SpShardedAuxHiddenStates
from .triton_adapters import register_deepseek_v41_kernels

logger = init_logger(__name__)

_INSTALLED = False


def _select_attn_cls(vllm_config) -> type[XcpuDeepseekV41Attention]:
    backend = vllm_config.attention_config.backend
    if backend is not None:
        raise ValueError(
            f"{backend.name} is not an XCPU attention backend; DeepSeek-V4.1 "
            "always runs XcpuDeepseekV41Attention on XCPU."
        )
    return XcpuDeepseekV41Attention


def _indexer_uses_fp4(vllm_config) -> bool:
    # Upstream also gates the MXFP4 indexer cache on SM10x GPUs. XCPU always
    # stores the 68 B MXFP4 key record; update_deepseek_v41_config has already
    # pinned indexer_kv_dtype to it.
    return True


def _use_v41_mxfp8_kv_record() -> bool:
    return True


def _insert_context_kv(
    attn: Any,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """DSpark context KV: ``_C.fused_deepseek_v4_kv_rope_insert`` is the KV
    half of the fused Q/KV insert, so it runs with zero query heads."""
    import torch_xcpu

    torch_xcpu.ops.deepseek_v41_qnorm_rope_kv_insert(
        kv.new_empty((kv.shape[0], 0, kv.shape[-1])),
        kv,
        attn.swa_cache_layer.kv_cache,
        slot_mapping,
        positions,
        attn.rotary_emb.cos_sin_cache,
        0,
    )


def _sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """``sp_reduce_scatter`` whose row padding stays in a compiled graph.

    Upstream pads only ``if sp_pad > 0``; the single TRACE_ONCE trace runs
    with an even token count and would drop the pad for odd ones. Padding by
    the symbolic remainder (zero rows when even) keeps every count valid.
    """
    # Reference: vLLM
    # vllm/models/common/ops/sequence_parallel.py::sp_reduce_scatter
    if not torch.compiler.is_compiling():
        return _upstream_sp_reduce_scatter(x)
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    x = F.pad(x, (0, 0, 0, (-x.shape[0]) % tp_size))
    output = sequence_parallel._custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return sequence_parallel.tensor_model_parallel_reduce_scatter(x, 0)


_upstream_sp_reduce_scatter = sequence_parallel.sp_reduce_scatter


def _gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """With Engram disaggregation, the one request per forward: the model
    forward calls this once on the full token stream (before sequence-parallel
    sharding) for every forward, including the dummy hashes of forwards
    without attention metadata."""
    # Reference: vLLM
    # vllm/models/deepseek_v41/common/engram.py::gather_engram_hashes
    client = get_engram_client()
    if client is None:
        return _upstream_gather_engram_hashes(
            hash_ids, dp_shared_memory=dp_shared_memory
        )
    client.request(hash_ids)
    return hash_ids


_upstream_gather_engram_hashes = nv_model.gather_engram_hashes


def _get_engram_dp_size() -> int:
    """DeepseekV4Model.forward feeds dummy hashes through gather_engram_hashes
    whenever this exceeds 1 (DP-sharded tables need every replica in the
    collective). The Engram exchange needs exactly that from every forward, so
    report a sharded table under disaggregation."""
    # Reference: vLLM
    # vllm/models/deepseek_v41/nvidia/model.py::DeepseekV4Model.forward
    if get_engram_client() is None:
        return _upstream_get_engram_dp_size()
    return 2


_upstream_get_engram_dp_size = nv_model.get_engram_dp_size


class _XcpuDeepseekV41LayerStack(nn.Module):
    """A range of the decoder layers as one compiled graph.

    The CUDA path keeps the model eager around breakable CUDA graphs; here the
    model forward stays eager (engram hashing, the decoder-replay sub-batch,
    sequence-parallel bookkeeping) and only the decoder layers are traced,
    with attention behind its opaque op. The layers are registered here as
    well as on the model so the graph owns their parameters; the model is held
    unregistered to keep the module tree acyclic.
    """

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", model: DeepseekV4Model
    ) -> None:
        super().__init__()
        self.layers = model.layers
        self.__dict__["model"] = model


@support_torch_compile
class _XcpuDeepseekV41SourceLayers(_XcpuDeepseekV41LayerStack):
    """Layers up to the last KV source; they always see the whole batch."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        # Reference: vLLM
        # vllm/models/deepseek_v41/nvidia/model.py::DeepseekV4Model._run_layers
        model: Any = self.model
        aux: dict[int, torch.Tensor] = {}
        if not model._defer_dspark_aux_sp_gather:
            outputs = DeepseekV4Model._run_layers(
                model,
                model.source_layer_ids,
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                aux,
                engram_hashes,
                engram_mask,
            )
            return (*outputs, *(aux[i] for i in model.source_aux_layer_ids()))

        mega_gate_metadata = model._mega_gate_metadata(input_ids)
        for idx in model.source_layer_ids:
            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = (
                model.layers[idx](
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    engram_hashes,
                    engram_mask,
                    capture_previous_aux=idx in model.aux_hidden_state_layers,
                    mega_gate_metadata=mega_gate_metadata,
                )
            )
            if previous_aux is not None:
                aux[idx] = previous_aux
        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        return (
            hidden_states,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            *(aux[i] for i in model.source_aux_layer_ids()),
        )


@support_torch_compile
class _XcpuDeepseekV41ReplayLayers(_XcpuDeepseekV41LayerStack):
    """Layers past the last KV source plus the collapse; in prefill steps the
    decoder replay runs them on each request's trailing window only."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Reference: vLLM
        # vllm/models/deepseek_v41/nvidia/model.py::DeepseekV4Model._run_replay_layers
        return DeepseekV4Model._run_replay_layers(
            self.model,
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
        )


class XcpuDeepseekV41Model(DeepseekV4Model):
    """``DeepseekV4Model`` with its decoder layers in compiled graphs."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        spec_config = vllm_config.speculative_config
        parallel_config = vllm_config.parallel_config
        self._pack_dspark_aux: bool = (
            spec_config is not None
            and spec_config.method == "dspark"
            and parallel_config.pipeline_parallel_size <= 1
            and parallel_config.prefill_context_parallel_size <= 1
        )
        self._defer_dspark_aux_sp_gather: bool = (
            self._pack_dspark_aux and self.use_sequence_parallel
        )
        self._packed_aux_buf: torch.Tensor | None = None
        self.source_layer_ids = range(self.start_layer, self.decoder_replay_start)
        self.source_layers = _XcpuDeepseekV41SourceLayers(
            vllm_config=vllm_config, prefix=f"{prefix}.source_layers", model=self
        )
        self.replay_layers = None
        if self.decoder_replay_layers is not None:
            self.replay_layers = _XcpuDeepseekV41ReplayLayers(
                vllm_config=vllm_config, prefix=f"{prefix}.replay_layers", model=self
            )

    def source_aux_layer_ids(self) -> list[int]:
        """The aux hidden states the source layers capture, in capture order:
        each layer's own post, so not the first layer's."""
        ids = self.source_layer_ids
        return [i for i in ids[1:] if i in self.aux_hidden_state_layers]

    def _run_layers(
        self,
        layer_ids: range,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ):
        if layer_ids != self.source_layer_ids:
            return super()._run_layers(
                layer_ids,
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                aux_hidden_by_layer,
                engram_hashes,
                engram_mask,
            )
        if (
            engram_hashes is None
            and self.engram_hash is not None
            and (
                not self.source_layers.do_not_compile
                or get_engram_client() is not None
            )
        ):
            # Profile and warmup runs hash nothing; the graph keeps the engram
            # layers anyway, fed dead n-grams whose gate is shut. ``positions``
            # holds the pre-``sp_shard`` token count (``input_ids`` is already
            # SP-sharded, or None when ``inputs_embeds`` is passed).
            engram_hashes, engram_mask = self.engram_hash.dummy_hashes(positions)
            gathered_hashes = _gather_engram_hashes(
                engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
            )
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                if layer.engram is not None:
                    layer.engram.prepare_embeddings(
                        gathered_hashes[:, layer.engram.layer_hash_index]
                    )
        outputs = self.source_layers(
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            engram_hashes,
            engram_mask,
        )
        aux_hidden_by_layer.update(zip(self.source_aux_layer_ids(), outputs[5:]))
        return outputs[:5]

    def _run_replay_layers(self, *inputs: torch.Tensor | None):
        assert self.replay_layers is not None
        return self.replay_layers(*inputs)

    def _collapse(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        if not self._defer_dspark_aux_sp_gather:
            return super()._collapse(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                aux_hidden_by_layer,
                full_num_tokens,
            )
        from vllm.distributed import tensor_model_parallel_all_reduce

        assert isinstance(hidden_states, torch.Tensor)
        if self.fuse_mhc_all_reduce:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = nv_model.mhc_post_tilelang(
            hidden_states, residual, post_mix, res_mix
        )
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_by_layer[self.end_layer] = hidden_states.mean(dim=1)
        return hidden_states

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> Any:
        out: Any = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
        if not self._pack_dspark_aux or not isinstance(out, tuple):
            return out
        hidden_states, aux_list = out
        if not aux_list:
            return out
        n_rows = aux_list[0].shape[0]
        total_dim = sum(t.shape[-1] for t in aux_list)
        if (
            self._packed_aux_buf is None
            or self._packed_aux_buf.shape[0] < n_rows
            or self._packed_aux_buf.shape[1] != total_dim
            or self._packed_aux_buf.dtype != aux_list[0].dtype
            or self._packed_aux_buf.device != aux_list[0].device
        ):
            self._packed_aux_buf = torch.empty(
                (n_rows, total_dim),
                dtype=aux_list[0].dtype,
                device=aux_list[0].device,
            )
        packed = self._packed_aux_buf[:n_rows]
        torch.cat(aux_list, dim=-1, out=packed)
        if self._defer_dspark_aux_sp_gather:
            return hidden_states, SpShardedAuxHiddenStates(
                packed_shard=packed,
                full_num_tokens=hidden_states.shape[0],
            )
        return hidden_states, packed


class XcpuDSparkDeepseekV41Model(DSparkDeepseekV4Model):
    """``DSparkDeepseekV4Model`` with stacked KV-only context projection and
    runtime idle-DP dummy short-circuits."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._context_wkv_linear: Any = None
        for layer in self.layers:
            proj = layer.attn.fused_wqa_wkv
            # Mark raw WKV row offset before submodule quant packing so
            # build_context_wkv_linear can initialize one stacked linear
            # without touching opaque packed weights.
            proj._xcpu_keep_raw_wkv_offset = layer.attn.q_lora_rank
            proj._xcpu_keep_raw_weight = True

    def build_context_wkv_linear(self) -> None:
        import torch_xcpu

        projs = [layer.attn.fused_wqa_wkv for layer in self.layers]
        offsets = [layer.attn.q_lora_rank for layer in self.layers]
        try:
            if not projs:
                return
            raw_fp8_list = [
                item
                for proj in projs
                if (item := getattr(proj, "_xcpu_raw_wkv_fp8", None)) is not None
            ]
            if len(raw_fp8_list) == len(projs):
                block_sizes = {item[2] for item in raw_fp8_list}
                scale_dtypes = {item[1].dtype for item in raw_fp8_list}
                if len(block_sizes) == 1 and len(scale_dtypes) == 1:
                    stacked_w = torch.cat(
                        [item[0] for item in raw_fp8_list], dim=0
                    ).contiguous()
                    stacked_s = torch.cat(
                        [item[1] for item in raw_fp8_list], dim=0
                    ).contiguous()
                    block_size = raw_fp8_list[0][2]
                    self._context_wkv_linear = torch_xcpu.ops.initialize_fp8_linear(
                        stacked_w, stacked_s, block_size
                    )
                    logger.info_once(
                        "DSpark context KV: stacked %d wkv layers into FP8 "
                        "linear (%s, backend=%s)",
                        len(projs),
                        tuple(stacked_w.shape),
                        self._context_wkv_linear.params.backend.name.lower(),
                    )
                    return

            raw_weights = [xcpu_raw_linear_weight(proj) for proj in projs]
            if all(
                isinstance(w, torch.Tensor)
                and w.ndim == 2
                and w.dtype == torch.bfloat16
                and w.device.type in ("mcpu", "privateuseone")
                and w.shape[0] > offset
                and getattr(proj, "bias", None) is None
                for w, proj, offset in zip(raw_weights, projs, offsets, strict=True)
            ):
                stacked_w = torch.cat(
                    [
                        w[offset:]
                        for w, offset in zip(raw_weights, offsets, strict=True)
                    ],
                    dim=0,
                ).contiguous()
                if torch_xcpu.ops.bf16_linear_shape_supported(
                    stacked_w.size(0), stacked_w.size(1)
                ):
                    self._context_wkv_linear = torch_xcpu.ops.initialize_bf16_linear(
                        stacked_w.detach(), None
                    )
                    logger.info_once(
                        "DSpark context KV: stacked %d wkv layers into BF16 "
                        "linear (%s, backend=%s)",
                        len(projs),
                        tuple(stacked_w.shape),
                        self._context_wkv_linear.params.backend.name.lower(),
                    )
        finally:
            for proj in projs:
                proj._xcpu_raw_wkv_fp8 = None
                proj._xcpu_raw_weight = None

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        import torch_xcpu

        if (
            aux_hidden_states.device.type in ("mcpu", "privateuseone")
            and torch_xcpu.ops.is_dummy_run()
        ):
            return aux_hidden_states.new_zeros((
                aux_hidden_states.shape[0],
                self.hidden_size,
            ))
        return super().combine_hidden_states(aux_hidden_states)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        import torch_xcpu

        if main_x.shape[0] == 0:
            return
        if (
            context_slot_mappings is None
            and main_x.device.type in ("mcpu", "privateuseone")
            and torch_xcpu.ops.is_dummy_run()
        ):
            return
        if self._context_wkv_linear is None:
            super().precompute_and_store_context_kv(
                main_x, context_positions, context_slot_mappings
            )
            return
        x = main_x if main_x.is_contiguous() else main_x.contiguous()
        all_kv = self._context_wkv_linear(x).view(
            main_x.shape[0], self.num_dspark_layers, self.config.head_dim
        )
        for i, (layer, kv) in enumerate(
            zip(self.layers, all_kv.unbind(1), strict=True)
        ):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)


def install_deepseek_v41() -> None:
    global _INSTALLED
    if _INSTALLED or current_platform.device_name != "mcpu":
        return

    verify_upstream_compatibility(("deepseek_v41",))
    register_deepseek_v41_kernels()
    rebind(vllm_ops, "topk_hash_softplus_sqrt", ops.topk_hash_softplus_sqrt)
    rebind(nv_model, "_select_dsv4_attn_cls", _select_attn_cls)
    rebind(nv_model, "mhc_pre_delayed_tilelang", ops.mhc_pre_delayed)
    rebind(nv_model, "mhc_post_tilelang", ops.mhc_post)
    rebind(nv_model, "mhc_shifted_post_pre", ops.mhc_shifted_post_pre)
    rebind(nv_model, "hc_collapse_triton", ops.hc_collapse)
    rebind(nv_model.DeepseekV41LLMForCausalLM, "model_cls", XcpuDeepseekV41Model)
    rebind(nv_model, "sp_reduce_scatter", _sp_reduce_scatter)
    rebind(nv_model, "gather_engram_hashes", _gather_engram_hashes)
    rebind(nv_model, "get_engram_dp_size", _get_engram_dp_size)
    rebind(nv_dspark, "DSparkDeepseekV4Model", XcpuDSparkDeepseekV41Model)
    rebind(nv_dspark, "mhc_post_tilelang", ops.mhc_post)
    rebind(nv_dspark, "hc_collapse_triton", ops.hc_collapse)
    rebind(nv_dspark, "_insert_context_kv", _insert_context_kv)
    rebind(v41_attention, "dsa_indexer_uses_fp4", _indexer_uses_fp4)
    rebind(v41_attention, "_use_v41_mxfp8_kv_record", _use_v41_mxfp8_kv_record)
    rebind(v41_attention, "DeepseekV4IndexerCache", XcpuDeepseekV41IndexerCache)
    rebind(v41_compressor, "fused_save_compress_norm", ops.fused_save_compress_norm)
    rebind(v41_compressor, "rope_quant_insert", ops.rope_quant_insert)
    from . import engram  # noqa: F401  (registers the OOT engram layers)

    _INSTALLED = True


install_deepseek_v41()


class XcpuDeepseekV41ForCausalLM(DeepseekV41ForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``language_model.*`` weights directly to the child loader
        instead of materializing and sorting all checkpoint tensors in memory."""
        # Reference: vLLM
        # vllm/models/deepseek_v41/nvidia/vl_model.py::DeepseekV41ForCausalLM.load_weights  # noqa: E501
        from vllm.model_executor.models.utils import (
            AutoWeightsLoader,
            StageMissingLayer,
        )

        child_modules = dict(self.named_children())

        def _grouped_stream() -> Iterable[tuple[str, torch.Tensor]]:
            deferred_non_lm: list[tuple[str, torch.Tensor]] = []
            for name, tensor in self.hf_to_vllm_mapper.apply(weights):
                if name.startswith("language_model."):
                    yield name, tensor
                else:
                    top_prefix = name.split(".", 1)[0]
                    if isinstance(child_modules.get(top_prefix), StageMissingLayer):
                        continue
                    deferred_non_lm.append((name, tensor.clone()))
            deferred_non_lm.sort(key=lambda x: x[0])
            yield from deferred_non_lm

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(_grouped_stream())
        self._weights_finalized = True
        return loaded_params


class XcpuDSparkDeepseekV41ForCausalLM(DSparkDeepseekV4ForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._loading_checkpoint_weights = True
        try:
            return super().load_weights(weights)
        finally:
            self._loading_checkpoint_weights = False

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        if not getattr(self, "_loading_checkpoint_weights", False) and hasattr(
            self.model, "build_context_wkv_linear"
        ):
            self.model.build_context_wkv_linear()
