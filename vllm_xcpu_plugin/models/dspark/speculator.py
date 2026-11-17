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
    """Route greedy DSpark argmax through ``torch_xcpu.ops.argmax_out`` and
    skip the sequential sampling head on runtime idle-DP dummy sync batches."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._sp_gather_buf: torch.Tensor | None = None

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
