# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch


def _xcpu_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype not in (None, torch.float8_e4m3fn):
        raise NotImplementedError(f"XCPU indexer quantization does not support {dtype}")
    if column_major_scales or tma_aligned_scales or out_q is not None:
        raise NotImplementedError(
            "XCPU indexer quantization only supports newly allocated row-major output"
        )

    import torch_xcpu

    return torch_xcpu.ops.per_token_group_quant_fp8(
        x,
        group_size,
        epsilon=eps,
        use_ue8m0=bool(use_ue8m0),
    )


def maybe_patch_indexer_quant() -> None:
    from vllm.model_executor.models import deepseek_v2

    module = cast(Any, deepseek_v2)
    if getattr(module, "_xcpu_indexer_quant_patched", False):
        return
    module.per_token_group_quant_fp8 = _xcpu_per_token_group_quant_fp8
    module._xcpu_indexer_quant_patched = True
