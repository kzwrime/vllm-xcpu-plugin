# SPDX-License-Identifier: Apache-2.0
"""XCPU overrides for the generic DSpark speculator."""

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

from vllm_xcpu_plugin.models.deepseek_v41.model import SpShardedAuxHiddenStates
from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility
from vllm_xcpu_plugin.upstream_contracts import rebind

_INSTALLED = False


class XcpuDSparkSpeculator(DSparkSpeculator):
    """XCPU DSpark speculator with fused Markov sampling, deferred SP aux gather,
    and idle-DP dummy short-circuits."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._d2t_target_ids: torch.Tensor | None = None
        self._sp_gather_buf: torch.Tensor | None = None
        self._confidence_markov_embeds: torch.Tensor | None = None

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
        can_use_fused = (
            base_logits.device.type in ("mcpu", "privateuseone")
            and base_logits.dtype == torch.bfloat16
            and not self.use_acceptance_estimator
            and self.draft_watermarker is None
            and (self.draft_logits is None or self._d2t_scatter_index is None)
            and lp_ok
        )
        resolved = (
            self._resolve_fused_markov_weights(vocab_size) if can_use_fused else None
        )
        if resolved is not None:
            from torch_xcpu import ops as xcpu_ops

            w1, w2_tensor, weight_layout = resolved
            probabilistic = self.draft_logits is not None
            markov_embeds_out: torch.Tensor | None = None
            if self.use_confidence_head:
                rank = w1.shape[1]
                if (
                    self._confidence_markov_embeds is None
                    or self._confidence_markov_embeds.shape[0] < num_sample
                    or self._confidence_markov_embeds.shape[1] != rank
                    or self._confidence_markov_embeds.device != base_logits.device
                ):
                    self._confidence_markov_embeds = torch.empty(
                        (max(num_sample, self.max_num_tokens), rank),
                        dtype=torch.bfloat16,
                        device=base_logits.device,
                    )
                markov_embeds_out = self._confidence_markov_embeds[:num_sample]

            xcpu_ops.dspark_markov_sample(
                self.draft_tokens[:num_reqs],
                base_logits,
                prev,
                w1,
                w2_tensor,
                draft_logits=self.draft_logits,
                markov_embeds_out=markov_embeds_out,
                d2t=self._d2t_target_ids if not probabilistic else None,
                idx_map=idx_map if probabilistic else None,
                sample_pos=sample_pos if probabilistic else None,
                temperature=self.temperature if probabilistic else None,
                seeds=self.seeds if probabilistic else None,
                weight_layout=weight_layout,
                probabilistic=probabilistic,
                use_fp64=self.use_fp64_gumbel,
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
