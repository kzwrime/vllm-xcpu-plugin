# SPDX-License-Identifier: Apache-2.0
"""XCPU overrides for the generic DSpark speculator."""

import os
from typing import Any

import torch
import vllm.v1.worker.gpu.sample.gumbel as gumbel_module
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as dspark_speculator
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_xcpu_plugin.models.deepseek_v41.sp_aux import SpShardedAuxHiddenStates
from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility
from vllm_xcpu_plugin.upstream_contracts import rebind

_INSTALLED = False

_MARKOV_FUSED_ENV = "VLLM_XCPU_DSPARK_MARKOV_FUSED"
_MARKOV_DTYPE_ENV = "VLLM_XCPU_DSPARK_MARKOV_DTYPE"
_MARKOV_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e4m3fn": "fp8",
    "fp8_ue8m0": "fp8",
    "int8": "int8",
    "int8_ue8m0": "int8",
}


def _dspark_markov_fused_enabled() -> bool:
    raw = os.environ.get(_MARKOV_FUSED_ENV, "1").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    raise ValueError(
        f"Unsupported {_MARKOV_FUSED_ENV}={raw!r}; "
        "expected one of: 1, 0, true, false, on, off, yes, no"
    )


def _configured_markov_dtype() -> str:
    raw = os.environ.get(_MARKOV_DTYPE_ENV, "bf16").strip().lower()
    mode = _MARKOV_DTYPE_ALIASES.get(raw)
    if mode is None:
        valid = ", ".join(sorted(_MARKOV_DTYPE_ALIASES))
        raise ValueError(
            f"Unsupported {_MARKOV_DTYPE_ENV}={raw!r}; expected one of: {valid}"
        )
    return mode


def _quantize_ue8m0(
    weight: torch.Tensor, mode: str, block_n: int = 32, block_k: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D [N, K] weight on CPU f32 with [block_n, block_k] UE8M0 scales."""
    n, k = weight.shape
    w_f32 = weight.detach().to(device="cpu", dtype=torch.float32)
    blocks = (
        w_f32.view(n // block_n, block_n, k // block_k, block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    amax = blocks.abs().amax(dim=(-2, -1))
    qmax = 448.0 if mode == "fp8" else 127.0
    raw_scale = amax / qmax
    mant, exp = torch.frexp(raw_scale)
    ceil_exp = torch.where(mant == 0.5, exp - 1, exp)
    s_int = torch.where(
        raw_scale > 0.0,
        (ceil_exp + 127).clamp(0, 254),
        torch.zeros_like(exp),
    )
    inv_scale = torch.ldexp(torch.ones_like(raw_scale), -(s_int.to(torch.int32) - 127))
    scaled = blocks * inv_scale.unsqueeze(-1).unsqueeze(-1)
    if mode == "fp8":
        q_blocks = scaled.to(torch.float8_e4m3fn)
    else:
        q_blocks = torch.round(scaled).clamp(-127, 127).to(torch.int8)
    q_row_major = q_blocks.permute(0, 2, 1, 3).contiguous().view(n, k)
    return q_row_major, s_int.to(torch.uint8).contiguous()


def _quantize_ue8m0_block32(
    weight: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_ue8m0(weight, mode, block_n=32, block_k=32)


def _quantize_ue8m0_block16(
    weight: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_ue8m0(weight, mode, block_n=16, block_k=16)


def _pack_quant_markov_w2(q_row_major: torch.Tensor, layout: int) -> torch.Tensor:
    """Pack row-major 1-byte quantized [N, K] weights to layout 0, 1, or 2."""
    n, k = q_row_major.shape
    if layout == 0:
        return q_row_major.contiguous()
    raw = q_row_major.contiguous().view(torch.uint8)
    if layout == 1:
        packed = (
            raw.view(n // 32, 32, k // 2, 2).permute(0, 2, 1, 3).contiguous().view(n, k)
        )
        return packed.view(q_row_major.dtype)
    if layout == 2:
        packed = raw.view(n // 32, 32, k).permute(0, 2, 1).contiguous().view(n, k)
        return packed.view(q_row_major.dtype)
    raise ValueError(f"Unsupported weight_layout={layout}")


def _layout_from_backend_name(backend_name: str | None) -> int:
    if backend_name == "INTEL_AMX":
        return 1
    if backend_name in ("PORTABLE", "PORTABLE_LUT", "ACC"):
        return 2
    return 0


class XcpuDSparkSpeculator(DSparkSpeculator):
    """XCPU DSpark speculator with fused Markov sampling, deferred SP aux gather,
    and idle-DP dummy short-circuits."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._d2t_target_ids: torch.Tensor | None = None
        self._sp_gather_buf: torch.Tensor | None = None
        self._confidence_markov_embeds: torch.Tensor | None = None
        self._quant_markov_cache: tuple[str, torch.Tensor, torch.Tensor, int] | None = (
            None
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        d2t = getattr(model, "draft_id_to_target_id", None)
        if d2t is not None:
            self._d2t_target_ids = (
                torch.arange(d2t.shape[0], dtype=torch.int64, device=d2t.device)
                + d2t.to(torch.int64)
            ).contiguous()
        else:
            self._d2t_target_ids = None
        self._quant_markov_cache = None
        if _dspark_markov_fused_enabled():
            mode = _configured_markov_dtype()
            if mode in ("fp8", "int8"):
                self._maybe_prepare_quant_markov_w2(mode, model)
        return model

    def _combine_target_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: (
            list[torch.Tensor] | torch.Tensor | SpShardedAuxHiddenStates | None
        ),
    ) -> torch.Tensor:
        if isinstance(aux_hidden_states, SpShardedAuxHiddenStates):
            local_main = self.model.combine_hidden_states(
                aux_hidden_states.packed_shard
            )
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                comm_ptr = getattr(
                    tp_group.device_communicator, "comm_ptr_wrapper", None
                )
                if (
                    local_main.device.type in ("mcpu", "privateuseone")
                    and comm_ptr is not None
                ):
                    from torch_xcpu import ops as xcpu_ops

                    local_main = local_main.contiguous()
                    padded_n = local_main.shape[0] * tp_group.world_size
                    hidden = local_main.shape[1]
                    if (
                        self._sp_gather_buf is None
                        or self._sp_gather_buf.shape[0] < padded_n
                        or self._sp_gather_buf.shape[1] != hidden
                        or self._sp_gather_buf.dtype != local_main.dtype
                        or self._sp_gather_buf.device != local_main.device
                    ):
                        self._sp_gather_buf = torch.empty(
                            (max(padded_n, self.max_num_tokens), hidden),
                            dtype=local_main.dtype,
                            device=local_main.device,
                        )
                    dst = self._sp_gather_buf[:padded_n]
                    xcpu_ops.all_gather_into_tensor_out_v2(
                        dst, local_main, comm_ptr, dim=0
                    )
                    return dst[: aux_hidden_states.full_num_tokens]
                return sp_all_gather(local_main)[: aux_hidden_states.full_num_tokens]
            return local_main[: aux_hidden_states.full_num_tokens]
        return super()._combine_target_hidden_states(
            last_hidden_states, aux_hidden_states
        )

    def _maybe_prepare_quant_markov_w2(
        self, mode: str, model: torch.nn.Module | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        draft_model = model if model is not None else getattr(self, "model", None)
        inner = getattr(draft_model, "model", None)
        markov_head = getattr(inner, "markov_head", None)
        if markov_head is None:
            return None
        w2_mod = getattr(markov_head, "markov_w2", None)
        if w2_mod is None or getattr(w2_mod, "bias", None) is not None:
            return None
        w2 = getattr(w2_mod, "weight", None)
        if not isinstance(w2, torch.Tensor) or w2.dim() != 2:
            return None
        vocab_size, rank = w2.shape
        if vocab_size <= 0 or rank <= 0:
            return None

        if mode == "int8":
            int8_linear = getattr(w2_mod, "_xcpu_int8_linear", None)
            if int8_linear is not None:
                params = int8_linear.params
                if params.bias is not None or tuple(params.scale_block_size) not in (
                    (16, 16),
                    (32, 32),
                ):
                    return None
                layout = _layout_from_backend_name(params.backend.name)
                if layout != 0 and (vocab_size % 32 != 0 or rank % 32 != 0):
                    return None
                w2_packed = params.packed_weight.detach()
                w2_scale = params.weight_scale.detach()
                self._quant_markov_cache = (mode, w2_packed, w2_scale, layout)
                return w2_packed, w2_scale, layout

        expected_dtype = torch.float8_e4m3fn if mode == "fp8" else torch.int8
        xcpu_linear = getattr(w2_mod, "_xcpu_bf16_linear", None)

        if w2.dtype == expected_dtype:
            scale = None
            for attr in ("markov_w2_scale", "weight_scale", "weight_scale_inv"):
                cand = getattr(w2_mod, attr, None)
                if cand is not None:
                    scale = cand
                    break
            if not isinstance(scale, torch.Tensor) or scale.dtype not in (
                torch.uint8,
                torch.float8_e8m0fnu,
            ):
                return None
            valid_block = False
            if mode == "fp8":
                valid_block = (
                    vocab_size % 32 == 0
                    and rank % 32 == 0
                    and tuple(scale.shape) == (vocab_size // 32, rank // 32)
                )
            else:
                valid_block = (
                    vocab_size % 16 == 0
                    and rank % 16 == 0
                    and tuple(scale.shape) == (vocab_size // 16, rank // 16)
                ) or (
                    vocab_size % 32 == 0
                    and rank % 32 == 0
                    and tuple(scale.shape) == (vocab_size // 32, rank // 32)
                )
            if not valid_block:
                return None
            if vocab_size % 32 != 0 or rank % 32 != 0:
                layout = 0
            elif xcpu_linear is not None:
                layout = _layout_from_backend_name(xcpu_linear.backend.name)
            else:
                from torch_xcpu import ops as xcpu_ops

                backend_cfg = (
                    xcpu_ops.configured_int8_linear_backend()
                    if mode == "int8"
                    else xcpu_ops.configured_bf16_linear_backend()
                )
                layout = _layout_from_backend_name(backend_cfg.name)
            w2_packed = (
                w2.contiguous()
                if layout == 0
                else _pack_quant_markov_w2(w2.detach().cpu(), layout).to(w2.device)
            )
            w2_scale = scale.detach().contiguous().to(w2.device)
            self._quant_markov_cache = (mode, w2_packed, w2_scale, layout)
            return w2_packed, w2_scale, layout

        if w2.dtype != torch.bfloat16:
            return None

        if mode == "fp8":
            if vocab_size % 32 != 0 or rank % 32 != 0:
                return None
            q_row_major, scale_u8 = _quantize_ue8m0_block32(w2, "fp8")
        else:
            if vocab_size % 16 != 0 or rank % 16 != 0:
                return None
            q_row_major, scale_u8 = _quantize_ue8m0_block16(w2, "int8")
        layout = (
            _layout_from_backend_name(xcpu_linear.backend.name)
            if xcpu_linear is not None
            else 0
        )
        if vocab_size % 32 != 0 or rank % 32 != 0:
            layout = 0
        w2_packed = _pack_quant_markov_w2(q_row_major, layout).to(w2.device)
        w2_scale = scale_u8.to(w2.device)
        self._quant_markov_cache = (mode, w2_packed, w2_scale, layout)
        if hasattr(w2_mod, "_xcpu_bf16_linear"):
            w2_mod._xcpu_bf16_linear = None
        if hasattr(w2_mod, "_xcpu_bf16_weight_key"):
            w2_mod._xcpu_bf16_weight_key = None
        return w2_packed, w2_scale, layout

    def _resolve_quant_markov_weights(
        self, vocab_size: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        inner = getattr(self.model, "model", None)
        markov_head = getattr(inner, "markov_head", None)
        if markov_head is None:
            return None
        w1_mod = getattr(markov_head, "markov_w1", None)
        if w1_mod is None:
            return None
        w1 = getattr(w1_mod, "weight", None)
        if (
            not isinstance(w1, torch.Tensor)
            or w1.dim() != 2
            or w1.dtype != torch.bfloat16
            or not w1.is_contiguous()
        ):
            return None
        rank = w1.shape[1]
        cache = getattr(self, "_quant_markov_cache", None)
        if cache is not None and cache[0] == mode:
            _, w2_packed, w2_scale, layout = cache
        else:
            prepared = self._maybe_prepare_quant_markov_w2(mode)
            if prepared is None:
                return None
            w2_packed, w2_scale, layout = prepared
        if (
            tuple(w2_packed.shape) != (vocab_size, rank)
            or w2_packed.device != w1.device
            or w2_scale.device != w1.device
        ):
            return None
        return w1, w2_packed, w2_scale, layout

    def _resolve_fused_markov_weights(
        self, vocab_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        inner = getattr(self.model, "model", None)
        markov_head = getattr(inner, "markov_head", None)
        if markov_head is None:
            return None
        w1_mod = getattr(markov_head, "markov_w1", None)
        w2_mod = getattr(markov_head, "markov_w2", None)
        if w1_mod is None or w2_mod is None:
            return None
        if getattr(w2_mod, "bias", None) is not None:
            return None
        w1 = getattr(w1_mod, "weight", None)
        if (
            not isinstance(w1, torch.Tensor)
            or w1.dim() != 2
            or w1.dtype != torch.bfloat16
            or not w1.is_contiguous()
        ):
            return None
        rank = w1.shape[1]
        xcpu_linear = getattr(w2_mod, "_xcpu_bf16_linear", None)
        if xcpu_linear is not None and vocab_size % 32 == 0 and rank % 2 == 0:
            backend_name = xcpu_linear.backend.name
            if backend_name == "INTEL_AMX":
                return w1, xcpu_linear.params.packed_weight, 1
            if backend_name in ("PORTABLE", "ACC"):
                return w1, xcpu_linear.params.packed_weight, 2
        w2 = getattr(w2_mod, "weight", None)
        if (
            isinstance(w2, torch.Tensor)
            and w2.dim() == 2
            and w2.dtype == torch.bfloat16
            and w2.shape == (vocab_size, rank)
        ):
            return w1, w2, 0
        return None

    def _alloc_confidence_embeds(
        self, num_sample: int, rank: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._confidence_markov_embeds is None
            or self._confidence_markov_embeds.shape[0] < num_sample
            or self._confidence_markov_embeds.shape[1] != rank
            or self._confidence_markov_embeds.device != device
        ):
            self._confidence_markov_embeds = torch.empty(
                (max(num_sample, self.max_num_tokens), rank),
                dtype=torch.bfloat16,
                device=device,
            )
        return self._confidence_markov_embeds[:num_sample]

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return
        if num_reqs <= 0:
            return

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        lp = getattr(self.model, "logits_processor", None)
        lp_ok = lp is None or (
            getattr(lp, "scale", 1.0) == 1.0 and getattr(lp, "soft_cap", None) is None
        )
        fused_enabled = _dspark_markov_fused_enabled()
        can_use_fused = (
            fused_enabled
            and base_logits.device.type in ("mcpu", "privateuseone")
            and base_logits.dtype == torch.bfloat16
            and not self.use_acceptance_estimator
            and self.draft_watermarker is None
            and (self.draft_logits is None or self._d2t_scatter_index is None)
            and lp_ok
        )
        markov_dtype = _configured_markov_dtype() if can_use_fused else "bf16"
        quant_resolved = (
            self._resolve_quant_markov_weights(vocab_size, markov_dtype)
            if (can_use_fused and markov_dtype in ("fp8", "int8"))
            else None
        )
        bf16_resolved = (
            self._resolve_fused_markov_weights(vocab_size)
            if (can_use_fused and quant_resolved is None)
            else None
        )
        if quant_resolved is not None or bf16_resolved is not None:
            from torch_xcpu import ops as xcpu_ops

            probabilistic = self.draft_logits is not None
            rank = (
                quant_resolved[0].shape[1]
                if quant_resolved is not None
                else bf16_resolved[0].shape[1]  # type: ignore[index]
            )
            markov_embeds_out = (
                self._alloc_confidence_embeds(num_sample, rank, base_logits.device)
                if self.use_confidence_head
                else None
            )
            common_kwargs: dict[str, Any] = {
                "draft_logits": self.draft_logits,
                "markov_embeds_out": markov_embeds_out,
                "d2t": self._d2t_target_ids if not probabilistic else None,
                "idx_map": idx_map if probabilistic else None,
                "sample_pos": sample_pos if probabilistic else None,
                "temperature": self.temperature if probabilistic else None,
                "seeds": self.seeds if probabilistic else None,
                "probabilistic": probabilistic,
                "use_fp64": self.use_fp64_gumbel,
            }
            if quant_resolved is not None:
                w1, w2_packed, w2_scale, weight_layout = quant_resolved
                op = (
                    xcpu_ops.dspark_markov_sample_fp8
                    if markov_dtype == "fp8"
                    else xcpu_ops.dspark_markov_sample_int8
                )
                op(
                    self.draft_tokens[:num_reqs],
                    base_logits,
                    prev,
                    w1,
                    w2_packed,
                    w2_scale,
                    weight_layout=weight_layout,
                    **common_kwargs,
                )
            else:
                assert bf16_resolved is not None
                w1, w2_tensor, weight_layout = bf16_resolved
                xcpu_ops.dspark_markov_sample(
                    self.draft_tokens[:num_reqs],
                    base_logits,
                    prev,
                    w1,
                    w2_tensor,
                    weight_layout=weight_layout,
                    **common_kwargs,
                )
            if self.use_confidence_head:
                assert markov_embeds_out is not None
                confidence = self.model.compute_confidence(
                    sample_hidden, markov_embeds_out
                )
                self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                    num_reqs, n_spec
                )
            return

        confidence_markov_embeds = []
        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i].add_(bias)
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            if (
                logits.device.type in ("mcpu", "privateuseone")
                and logits.dtype == torch.bfloat16
                and logits.dim() == 2
                and logits.is_contiguous()
                and logits.size(1) > 0
                and logits.size(1) % 2 == 0
                and (logits.data_ptr() & 3) == 0
            ):
                from torch_xcpu import ops as xcpu_ops

                output = torch.empty(
                    logits.size(0), dtype=torch.int64, device=logits.device
                )
                draft_ids = xcpu_ops.argmax_out(logits, -1, False, output)
            else:
                draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        sampler = (
            gumbel_module.gumbel_sample
            if self.draft_watermarker is None
            else self.draft_watermarker.sample
        )
        return sampler(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if cudagraph_runtime_mode == CUDAGraphMode.NONE and head_hidden.device.type in (
            "mcpu",
            "privateuseone",
        ):
            from torch_xcpu import ops as xcpu_ops

            if xcpu_ops.is_dummy_run():
                return
        self._sample_sequential(num_reqs, head_hidden)


def install_dspark() -> None:
    global _INSTALLED
    if _INSTALLED or current_platform.device_name != "mcpu":
        return
    verify_upstream_compatibility(("dspark",))
    rebind(dspark_speculator, "DSparkSpeculator", XcpuDSparkSpeculator)
    _INSTALLED = True
