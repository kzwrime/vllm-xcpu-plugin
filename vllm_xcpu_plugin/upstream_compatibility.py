# SPDX-License-Identifier: Apache-2.0
"""Fail-closed checks for vLLM operators replaced outside Fake Triton.

These targets never reach :class:`FakeJITFunction`: the plugin replaces a
wrapper, custom-op implementation, or attention backend before the upstream
Triton launch happens.  They still need the same source-drift protection as
Fake Triton registrations because the XCPU implementation mirrors their
semantics.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm_xcpu_plugin.fake_triton.runtime import (
    KernelVersionError,
    _signature_fingerprint,
    _source_fingerprint,
)


@dataclass(frozen=True)
class UpstreamOperator:
    category: str
    module: str
    name: str
    source_version: str
    expected_source_hash: str
    expected_signature_hash: str
    replacement: str

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


def _operator(
    category: str,
    module: str,
    name: str,
    source_hash: str,
    signature_hash: str,
    replacement: str,
    *,
    source_version: str,
) -> UpstreamOperator:
    return UpstreamOperator(
        category=category,
        module=module,
        name=name,
        source_version=source_version,
        expected_source_hash=source_hash,
        expected_signature_hash=signature_hash,
        replacement=replacement,
    )


# Every entry owns its manually audited source version.  Do not infer this
# field from Git topology or replace it with a package-wide version: XCPU vLLM
# branches are assembled by cherry-pick and manual porting, and are not linear.
UPSTREAM_OPERATORS: tuple[UpstreamOperator, ...] = (
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "_cast_kv_tile",
        "02ad4f8276df109a75891591092dab5a2d7edacaa58a83a3bbfd49e1d7f5d133",
        "e35a5687b0f95309deb2a2601c82baad1a95d6475d6b7d5be0bc46c76e8f8379",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "_load_q_td",
        "8771a400987afae26a1703c79b56e9c7aafb57f10ccd22e439ae04ba95caf92a",
        "ffe36a9186a7ec4828182d97f67f93ab49eb5e45a17f71184972e2b8a8db5aaf",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "_load_kv_tile_td",
        "1c8bcd2152e7625a2b7b787a5679e85ba2c0eec10aed23bc5524bfb84df08ccf",
        "e577ef43e805008f988a6ed44578f5622a1d3cbad0df64bca547df4f59109b9f",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "_store_output_td",
        "ad0ee877daea85812fc980120edc0dca7d509419055a991aab3ae68f621c27ff",
        "4216f472eae74bf8c86e8b7543ea188c18400d34701bb7108269e8289c964e8d",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "kernel_unified_attention",
        "16c1a25c5b632c385717cf8d76b3bdc71abc1c2012b925f2a1e427bdf2d42275",
        "edefe3f15b40eca1cd7814d025601582962cc7436639f0d0431a45a51544c30e",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.25.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "reduce_segments",
        "e962b226a8cd8f09b5508028f6a397aaadc28445b5d2f99686fe1e740d94def9",
        "e72050c4e94145a035b30d2200f0343dec536ca87af6ecd9f61bc9ac793e9126",
        "torch_xcpu.ops.unified_attention",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
        "reshape_and_cache_kernel_flash",
        "cac1bb4e4f729f4d67a964be3a3121f0a1f19f2f9074e7d05a1ef343e29b2b48",
        "56bb05305325dd4587de35de5535bc81a8dcddc5b5970d42b2267427f464050b",
        "torch_xcpu.ops.reshape_and_cache",
        source_version="v0.24.0",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
        "_reshape_cache_per_token_head",
        "2e3a46589abe8bc76351690b945d061da648f9c7954cad997ec444840384b950",
        "2c73d4f99ef1d3bac7037974d530da022b16fe903ee9ce30a54241a75cfa032c",
        "torch_xcpu.ops.reshape_and_cache",
        source_version="v0.24.0",
    ),
    # v0.25 added an explicit head stride for packed/strided diff-KV cache
    # views. XCPU backends still publish their platform-specific legacy cache
    # layouts; this audit does not claim packed diff-KV layout support.
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
        "reshape_and_cache_kernel_flash_diffkv",
        "2c5205b8393f541e9b0238b93622347edf01880623be5632bf0d0cce464be4ff",
        "feac593fbac5937165ee6fae9e806f8e7603a607d75acfbaad8562f9a3cd09a3",
        "torch_xcpu.ops.reshape_and_cache",
        source_version="v0.25.0",
    ),
    _operator(
        "conv",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "_causal_conv1d_fwd_kernel",
        "07d3f9c5973a7a6ffa11855e1da5a08d533e1025ff3de4209a2ce3f2e5aabdc2",
        "b5c2b2c03a2741d023b970d6805dfb9b84d12ba7386a681376e5b93f15c6c550",
        "torch_xcpu.ops.causal_conv1d_fn",
        source_version="v0.24.0",
    ),
    _operator(
        "conv",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "_causal_conv1d_update_kernel",
        "bf870fd5d0032a8fe09556972324ebca6e6ff776e70e6267f76615fa4ede5794",
        "60df9437065d228e91b9c1cec9d83fbdafa3a5f5fdd98e24a381a1d831b5ba43",
        "torch_xcpu.ops.causal_conv1d_update",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "fused_gdn_gating_kernel",
        "faecbca9b27a358972320a9f675f9bd596dfd8af4719e88bf1f431abd1199e3e",
        "aec37b8b6a0ec543380b81ff2718fa94fdabd9ec362dc3bb639244cc3aa3ac35",
        "torch_xcpu.ops.fused_gdn_gating",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv",
        "_fused_post_conv_kernel",
        "620d19949a5dffb673d24ea9c1d9b74db8a128388f117503c2896890151b7653",
        "208cfec03a7ca91a4a8d542509d13573afce9e6381c51c1a22505c2e417dda72",
        "torch_xcpu.ops.fused_post_conv_prep",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating",
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "cdc9983c0408370e378301a04da150046cadc779ad03ac6a3e49bdbb7056edfe",
        "feb78517f3a3980c55f16745ecd2176366adf5dee240ef8449c7071078efbd0d",
        "torch_xcpu.ops.fused_sigmoid_gating_delta_rule_update",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode_kernel",
        "287b5c77063ef335b8c709f0fa215d9e6285523b028d396babd54f7470fc9ed2",
        "767617c77c93207a7a4a28b9a74ab231aef7fcac82be83a159326610f4a0f854",
        "torch_xcpu.ops.fused_recurrent_gated_delta_rule_packed_decode",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "ChunkGatedDeltaRule.forward_native",
        "8c43190ae8408d298dd17a97245cf739e12a87cfb5ea910f0612da85d6780e4c",
        "9add943c04e663bf253f324ca048fddb84ba86fb9034e0acf64156aa064d57c4",
        "torch_xcpu.ops.chunk_gated_delta_rule_separated",
        source_version="v0.24.0",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.fla.ops.chunk",
        "chunk_gated_delta_rule",
        "3085ebdd828a04091edf37ad90cd724c12f4fb712526e4bf9d13806864689942",
        "9b0a1094cc96cc1c9609f4b02d8742163e2b8b819f4708d4ab4935fab9b41295",
        "torch_xcpu.ops.chunk_gated_delta_rule_separated",
        source_version="v0.24.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "_temperature_kernel",
        "803856df45907bd0e0d587c4e43ffd5e2d922e453dd992d3abe45e660e22355d",
        "f5a324a7ef2086943852131768de619f8ca9560a15bf3cd5050aa2c3cd6f2b2f",
        "torch_xcpu.ops.vllm_temperature_kernel",
        source_version="v0.24.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "tl_rand64",
        "6f240deb8a7d67d9250ffd35d62beef6300a7745091e02d715376655ee039cc7",
        "6223b23734cfe8824387f0f645e1001e708358f46915882633790b0c97e1c6d7",
        "torch_xcpu.ops.gumbel_sample",
        source_version="v0.24.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "tl_rand32",
        "1e313fec8ffef8569eba06cc0d59361c57af4169069d59c022c427dc594733a6",
        "6223b23734cfe8824387f0f645e1001e708358f46915882633790b0c97e1c6d7",
        "torch_xcpu.ops.gumbel_sample",
        source_version="v0.24.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "gumbel_block_argmax",
        "3d1a2df4886837e9ffb79610021e8e9df19d185602488db43dcfaab15c8f6c7f",
        "c8860dbf8e5f8870e636e585b95b2d98d7dbdb5506f7fe004fca10cc0c9ccc42",
        "torch_xcpu.ops.gumbel_sample",
        source_version="v0.25.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "_gumbel_sample_kernel",
        "0d90d2fa7965fcd6461e7c3c49031715e704b80c097aad7c344a772c7b5b0833",
        "e8b4d1955ce23439792abed60f92bd1a9943db262ba7c9a5f213d89eea348278",
        "torch_xcpu.ops.gumbel_sample",
        source_version="v0.24.0",
    ),
    _operator(
        "topk_topp",
        "vllm.v1.sample.ops.topk_topp_triton",
        "_update_min_larger_stats",
        "64187be916f8ab78f72f8b346c525b91d70e9e79676487dc4ac1665004e6919d",
        "721565a9b8cfd07968efb7445ed64fe86a7dace1c8dbb9633e8e8e9a1f3e091c",
        "torch_xcpu.ops.apply_top_k_top_p",
        source_version="v0.24.0",
    ),
    _operator(
        "topk_topp",
        "vllm.v1.sample.ops.topk_topp_triton",
        "_topk_topp_kernel",
        "4d4f1dbd297ca5b4b883989c427d45a65f79594bf8c7179151fce1dc27243126",
        "cf11f86d0b32b1dcc2b5cb4cb7e5d4ee0623496751af6731e961e402abf9718c",
        "torch_xcpu.ops.apply_top_k_top_p",
        source_version="v0.24.0",
    ),
    _operator(
        "grouped_topk",
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
        "grouped_topk",
        "25f175864bd6fa4cf210b00a99810304ac434b1d7bd16ede8fb92481cf62432b",
        "4f7fbcf35818ef44b6a629633d8d8145be49b28e7bdb7380e59600957800964f",
        "torch_xcpu.ops.grouped_topk",
        source_version="v0.19.0",
    ),
    _operator(
        "topk_softmax",
        "vllm._custom_ops",
        "topk_softmax",
        "008c94acb29fefc4c51cfe8e66f4a1648fec11f26e19835a97869ee03b1c336b",
        "80b417a27ce012f9dc87534716e4e18d2630b91cc13382c677a3e8fcf24460dc",
        "torch_xcpu.ops.topk_softmax",
        source_version="v0.25.0",
    ),
)


def _underlying_function(obj: Any) -> Any:
    # FakeJITFunction stores the undecorated Python function in ``fn``.  Real
    # Triton JITFunction exposes the same attribute in supported vLLM builds.
    return getattr(obj, "fn", obj)


def _resolve_target(module_name: str, attribute_path: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        obj = getattr(obj, component)
    return obj


def _mismatch_message(
    target: UpstreamOperator,
    *,
    actual_source_hash: str,
    actual_signature_hash: str,
) -> str:
    changed = []
    if actual_source_hash != target.expected_source_hash:
        changed.append(
            "source hash "
            f"expected {target.expected_source_hash}, got {actual_source_hash}"
        )
    if actual_signature_hash != target.expected_signature_hash:
        changed.append(
            "signature hash "
            f"expected {target.expected_signature_hash}, got {actual_signature_hash}"
        )
    return (
        f"{target.qualname}: upstream operator compatibility mismatch "
        f"({'; '.join(changed)}). "
        f"The XCPU replacement {target.replacement} was reviewed against "
        f"{target.source_version}. "
        "Manually review the upstream source diff and semantic changes, update the "
        "XCPU implementation and differential tests as needed, then update this "
        "operator's expected hashes and source_version in "
        "vllm_xcpu_plugin.upstream_compatibility."
    )


def verify_upstream_compatibility(
    categories: Iterable[str] | None = None,
) -> tuple[UpstreamOperator, ...]:
    """Validate all selected non-Fake-Triton replacement dependencies."""
    selected = set(categories) if categories is not None else None
    checked = []
    for target in UPSTREAM_OPERATORS:
        if selected is not None and target.category not in selected:
            continue
        try:
            obj = _resolve_target(target.module, target.name)
        except (ImportError, AttributeError) as exc:
            raise KernelVersionError(
                f"{target.qualname}: upstream operator is unavailable. The XCPU "
                f"replacement {target.replacement} was reviewed against "
                f"{target.source_version}. "
                "Manually inspect the upstream move/removal and update the "
                "replacement, tests, and per-operator compatibility manifest."
            ) from exc

        fn = _underlying_function(obj)
        actual_source_hash = _source_fingerprint(fn)
        actual_signature_hash = _signature_fingerprint(inspect.signature(fn))
        if (
            actual_source_hash != target.expected_source_hash
            or actual_signature_hash != target.expected_signature_hash
        ):
            raise KernelVersionError(
                _mismatch_message(
                    target,
                    actual_source_hash=actual_source_hash,
                    actual_signature_hash=actual_signature_hash,
                )
            )
        checked.append(target)
    return tuple(checked)
