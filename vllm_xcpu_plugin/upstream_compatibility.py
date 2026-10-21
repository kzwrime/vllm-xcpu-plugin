# SPDX-License-Identifier: Apache-2.0
"""Fail-closed checks for vLLM operators replaced outside Fake Triton.

These targets never reach :class:`FakeJITFunction`: the plugin replaces a
wrapper, custom-op implementation, or attention backend before the upstream
Triton launch happens.  They still need the same source-drift protection as
Fake Triton registrations because the XCPU implementation mirrors their
semantics.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
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
    alternate_signature_hashes: tuple[str, ...] = ()

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(frozen=True)
class UpstreamSourceFile:
    category: str
    module: str
    expected_sha256: str


# v0.31 audit: uncompressed FP8 indexer cache, BF16 Q and C++ TopK.
# New candidate filtering, FP4 cache, CP, speculative varlen decode and HiSparse
# are explicitly rejected. CPU sequence lengths are upper bounds, not masks.
# Sparse MLA consumes logical indices on the producer stream; no CUDA side-stream
# physical-index conversion is used. The outer wrapper retains shared TopK layers.
UPSTREAM_SOURCE_FILES: tuple[UpstreamSourceFile, ...] = (
    UpstreamSourceFile(
        "sparse_indexer",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "2282955cf1612f0b46110368fdbcefce60e11bb36204ad5801da47940ca33741",
    ),
    UpstreamSourceFile(
        "sparse_indexer",
        "vllm.v1.attention.backends.mla.indexer",
        "e601dd278f70d36519e12ccd52aa9de9c85ceafb16ba42e7bdd561110a0b41d9",
    ),
    UpstreamSourceFile(
        "sparse_indexer",
        "vllm.model_executor.models.deepseek_v2",
        "d547348698a5bab40d6d646213c8dc5fae9e8d98de8d2359dfec45d92e81c1de",
    ),
    UpstreamSourceFile(
        "sparse_mla",
        "vllm.model_executor.layers.mla",
        "eb159d09c8d00f7ef06ce9cd50821ea0cd54bcfd216d4d5b87d3a8522ea93e89",
    ),
    UpstreamSourceFile(
        "sparse_mla",
        "vllm.model_executor.layers.attention.mla_attention",
        "cdaae452880eb06a0c62835c490756e3b14c5827e85e0d3705fbb93bbb12e63b",
    ),
    UpstreamSourceFile(
        "sparse_mla",
        "vllm.model_executor.layers.attention.sparse_mla_attention",
        "59b5cc01f45c9a843db53413fcf970acb90d14cef0282115276b055b98d282d2",
    ),
)


def _operator(
    category: str,
    module: str,
    name: str,
    source_hash: str,
    signature_hash: str,
    replacement: str,
    *,
    source_version: str,
    alternate_signature_hashes: tuple[str, ...] = (),
) -> UpstreamOperator:
    return UpstreamOperator(
        category=category,
        module=module,
        name=name,
        source_version=source_version,
        expected_source_hash=source_hash,
        expected_signature_hash=signature_hash,
        replacement=replacement,
        alternate_signature_hashes=alternate_signature_hashes,
    )


# Every entry owns its manually audited source version.  Do not infer this
# field from Git topology or replace it with a package-wide version: XCPU vLLM
# branches are assembled by cherry-pick and manual porting, and are not linear.
UPSTREAM_OPERATORS: tuple[UpstreamOperator, ...] = (
    # Guard the Python interfaces as well as the Triton kernels: XCPU bypasses
    # unified_attention and forwards constructor arguments to the upstream base.
    # Audited v0.31 defaults: decoder, unquantized KV and ordinary sliding window;
    # pinning the full wrapper does not claim support for its optional features.
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "unified_attention",
        "61d44737cbe8baa514d5c7f198d2e91ae49820be46dc3671c1a5a29350074921",
        "910052c0787dc824128db7e6d1229e3a4b80ab18cb25fc02e83430424f7d0876",
        "torch_xcpu.ops.unified_attention",
        source_version="9c77da8aee (v0.31) attention Python interfaces",
    ),
    _operator(
        "attention",
        "vllm.v1.attention.backends.triton_attn",
        "TritonAttentionImpl.__init__",
        "76ac93b2d2dc22ade6a39d20e3e3f937dab7a8410a9c6396cdd74d05974a6ac9",
        "7beb5deda87f48833ea846faf5de980247c23c273e7e4a77d88fe02d22edee31",
        "vllm_xcpu_plugin.attn_backend.XcpuTritonAttentionImpl.__init__",
        source_version="9c77da8aee (v0.31) attention Python interfaces",
    ),
    _operator(
        "sparse_indexer",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "sparse_attn_indexer",
        "73eb3bcba9993f6e03398956a4b2776f3b3061dc63484d73f78afc3cc5f61e13",
        "6fc5720ed074bf65c06c5b66e3ee6382da2e45613a3d20fb9087320aebd223b6",
        "torch.ops.torch_xcpu.sparse_attn_indexer",
        source_version=(
            "v0.31 GLM5.2 PCP=DCP=1, uncompressed FP8 K, C++ indexer 2026-10-14"
        ),
        # LayerNameType resolves to str when VLLM_USE_LAYERNAME=0 and to
        # LayerName otherwise. Both are the same source-level annotation.
        alternate_signature_hashes=(
            "be5c410bd89f2649615b5c932c6a9ac672ae04255a758574a7d86ed61f9bd534",
        ),
    ),
    _operator(
        "sparse_indexer",
        "vllm.v1.attention.backends.utils",
        "split_decodes_and_prefills",
        "21b78ee405f260ca1faf720cac023103e278d378e45b2610f35c99bc265820e1",
        "441a21b245c30963704e4bcc4aa2a1aa4c994b575954c3318efa2b3d045b08e5",
        "torch.ops.torch_xcpu.sparse_attn_indexer",
        source_version=(
            "v0.31 GLM5.2 PCP=DCP=1, uncompressed FP8 K, C++ indexer 2026-10-14"
        ),
    ),
    _operator(
        "attention",
        "vllm.v1.attention.ops.triton_attention_helpers",
        "compute_kv_seq_mask",
        "69105e38cf99abb11f70ea627ade1c212c5186cc0debb75e79de7c839746171b",
        "db24d71f29cbc3ce0c7fd5347329e17dfb8a6568d31dad3543a166b79e26ee71",
        "torch_xcpu.ops.unified_attention",
        source_version=(
            "9c77da8aee (v0.31) dense causal/noncausal sliding-window mask; "
            "no PrefixLM, chunked attention or R-SWA support claimed"
        ),
    ),
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
        "af801ecf9d243a25e5d0390e5daff62bd24634aed22da0298a8ba381e057d350",
        "edefe3f15b40eca1cd7814d025601582962cc7436639f0d0431a45a51544c30e",
        "torch_xcpu.ops.unified_attention",
        source_version=(
            "v0.31 dense causal attention; local=dd24619b0d47831e; "
            "upstream=07ea9350baf84e33,1117140edbaf3d1e"
        ),
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
        "4b55f6864f0a931b431eb9109fc37cc7ac6cc2211f978f3695a1451bfece63fc",
        "919094d32027d86f0d85afb7297d54e3df4a029145fea3be4f81cb3cce8901f1",
        "torch_xcpu.ops.causal_conv1d_fn",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "conv",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "_causal_conv1d_update_kernel",
        "319484f5441cec6dad37ef6654f83c42e23635cbcf2c265c51882c24b170ef22",
        "2ce2444a568f366aea01e2eb2d695a5ecf2834188ebe9c6111bc992d9823412f",
        "torch_xcpu.ops.causal_conv1d_update",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating",
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "cdc9983c0408370e378301a04da150046cadc779ad03ac6a3e49bdbb7056edfe",
        "feb78517f3a3980c55f16745ecd2176366adf5dee240ef8449c7071078efbd0d",
        "torch_xcpu.ops.fused_sigmoid_gating_delta_rule_update",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode_kernel",
        "4356e2d51203a974057711776f2d3594b1b8e9874bebd6033d5ffa378f4b7ccc",
        "d1f29a4ace8c7ed438b13f2ae107e669d379aa968312e549c7d4a447ba4810f1",
        "torch_xcpu.ops.fused_recurrent_gated_delta_rule_packed_decode",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv",
        "_fused_post_conv_kernel",
        "620d19949a5dffb673d24ea9c1d9b74db8a128388f117503c2896890151b7653",
        "208cfec03a7ca91a4a8d542509d13573afce9e6381c51c1a22505c2e417dda72",
        "torch_xcpu.ops.fused_post_conv_prep",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "ChunkGatedDeltaRule.forward_native",
        "8c43190ae8408d298dd17a97245cf739e12a87cfb5ea910f0612da85d6780e4c",
        "9add943c04e663bf253f324ca048fddb84ba86fb9034e0acf64156aa064d57c4",
        "torch_xcpu.ops.chunk_gated_delta_rule_separated",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.third_party.flash_linear_attention.ops.chunk",
        "chunk_gated_delta_rule",
        "3085ebdd828a04091edf37ad90cd724c12f4fb712526e4bf9d13806864689942",
        "9b0a1094cc96cc1c9609f4b02d8742163e2b8b819f4708d4ab4935fab9b41295",
        "torch_xcpu.ops.chunk_gated_delta_rule_separated",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.v1.worker.mamba_utils",
        "_copy_mamba_state_block",
        "7c3f4734b6dfca5d8c82412152b15285ad01aabf536547fc4621924c3da9816a",
        "0dfbce2c07b466a0e091a79214c8bb6b6627e8b30b462b27c281f4c5ade60ed3",
        "torch.ops.mcpu.vllm_precopy_mamba_align / vllm_postprocess_mamba",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn",
        "vllm.v1.worker.mamba_utils",
        "_memcpy_u64_tiled",
        "7c118ce5ce4c75737552f6bdf5c4b01d0f7704128c244e9d5a1ff0ba90c2c3b9",
        "431ca3b125ad1346703a29bd33f913ef50678ddf54e45d42ab6f77821af308b2",
        "native memmove in mamba state copy",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "conv",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "causal_conv1d_update",
        "a511c321a827ed9bc195088f65e830b75916cb72d19ccd899a84e0842f6fe137",
        "f2fee20e91819d9550347e1a88e4bf77c25a982be633ee0ea2f326d165b0acd6",
        "torch_xcpu.ops.causal_conv1d_update",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    # Metadata build and the helpers mirrored by the fused C++ implementation.
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadata",
        "7817286309cfac315f4b8b717dcedab2537eebf18634bea35e353820a493afda",
        "127002ec8cee2a101275c778290c778184b1fcfe6ae630ac01b966b21bd60e0f",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder.__init__",
        "9bcd86391de6b79fd5cac69666f784cc596b7088eba87227d720cf568ecc3cef",
        "e73d1aaed9c63b280614c90e3cfe174fae3c007fbe68b0275f223e7868700961",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder.build",
        "f94a339faa63d868b75c9640eaa981f3410b526177c2399b241462fd023d1f98",
        "6a9c419a47084c40387d3cf2cbdb4ea3bc55a2a15a4a237dab55d3e71585ea5a",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder._build_chunk_metadata",
        "09b2029d1c6c376c06b96765d07a56071833ea61d30ca23406353b67344d8023",
        "07d60cbe3244c13eaa846c5b66127d751cb6cd679789cf4e235030a89bfce9c4",
        "torch_xcpu::build_gdn_metadata_out + _xcpu_gdn_metadata_build",
        source_version="v0.31; local=b7c90492ea; metadata audit 2026-10-14",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder._stage_spec_decode",
        "ed4a0dd7f58ffcbe25635000546766dfb3858763a6c4d1c6037095e78fbd6840",
        "4b805450702ff3f5ca3442c702399be4225a1774c5b64383ce8620a473d159e9",
        "torch_xcpu::build_gdn_metadata_out + _xcpu_gdn_metadata_build",
        source_version="v0.31; local=b7c90492ea; metadata audit 2026-10-14",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder._stage_decode",
        "37f208c81ed55c55e6f22e5eb75f6eff62013106210331f5775d6c6bbbac9da6",
        "5f4fe1c0bb533573eb6cd42779720b3e9ab6fc16016725598299c50a1baa70a4",
        "torch_xcpu::build_gdn_metadata_out + _xcpu_gdn_metadata_build",
        source_version="v0.31; local=b7c90492ea; metadata audit 2026-10-14",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadataBuilder.build_for_cudagraph_capture",
        "7b5c7ce6933d3622f586f710ad1f3a27d084e05342409933271fee1e888d8ffd",
        "c1198accfb83b14e33c7738dacdca0171a94fd3a226e6936d236407e37dbb65e",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backend",
        "CommonAttentionMetadata.compute_num_computed_tokens",
        "959fb6da987d8ba69582540524655be7d014c9ce8c7ed533e9a68ace578a1812",
        "d66922ee7dbd1de1830c2c0a26b18b8a5fe42592c8d1d7bf09b3dcdf0b175ee1",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.utils",
        "split_decodes_and_prefills",
        "21b78ee405f260ca1faf720cac023103e278d378e45b2610f35c99bc265820e1",
        "441a21b245c30963704e4bcc4aa2a1aa4c994b575954c3318efa2b3d045b08e5",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.utils",
        "mamba_get_block_table_tensor",
        "b9f8c8a73cfd9f9ca9810155e38bf95ad780f7141dba5972965a59fd000b7656",
        "ba09cc29df138885574deba02b9e4c00ed37b29eb4c24b79feaebaef648c288f",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.v1.attention.backends.utils",
        "compute_causal_conv1d_metadata",
        "1554acd8bc1eb87695f895ebae0fa07780043155803edb34cd357b84e91466e4",
        "4fa5cf3493cf2874dc51cd4f3c311ff123d4327e19902bb68cfde506cbf47f36",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.third_party.flash_linear_attention.ops.index",
        "prepare_lens",
        "2cbb1a08d9fa58701823214a0d3f9916b4fb1fa64e49ff2eb7919e593e8dd10b",
        "32ad9d1f25ddc72ee74d78eef7f066bef049be5efeeecce9dc90178ec3fefcd3",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.third_party.flash_linear_attention.ops.index",
        "prepare_chunk_indices",
        "5570ba010b766321161b897258db6a2d6b4bcd14a804e2e8de723b4b902aea64",
        "af2e8eeda7ac8469d8e728556656e91733866950690f917c106fc734633ec3e3",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
    ),
    _operator(
        "gdn_metadata",
        "vllm.third_party.flash_linear_attention.ops.index",
        "prepare_chunk_offsets",
        "eef70582e66dc90b622986741d57aac543f8e37c28ed22e5c188bdfa5d47771c",
        "af2e8eeda7ac8469d8e728556656e91733866950690f917c106fc734633ec3e3",
        "torch_xcpu::build_gdn_metadata_out",
        source_version="v0.31; local=630fdc2cf5; Qwen3.5 semantic review",
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
        "tl_rand32",
        "1e313fec8ffef8569eba06cc0d59361c57af4169069d59c022c427dc594733a6",
        "6223b23734cfe8824387f0f645e1001e708358f46915882633790b0c97e1c6d7",
        "torch_xcpu.ops.gumbel_sample",
        source_version="v0.24.0",
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "gumbel_noised_argmax",
        "e59dd555ff5fbff70ad0ffe15df709364f7b47715d99958e11892b7f1ab326d8",
        "629ea9a6ef230122b37ca719a0d28743bfcb263e22aab0feadf05af08f1c30b6",
        "torch_xcpu.ops.gumbel_sample",
        source_version=(
            "v0.31 XCPU RNG; local=dd24619b0d47831e; upstream=18553c5c9ff02acb"
        ),
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "gumbel_block_argmax",
        "84cb0d04c7c15e3065ca112dc9a757268b093a63c4a5e32b5a53f14ff533fd4d",
        "6b3bee3b297aa826a02b3c994713dcfd70ab3a48eaa1a49dad5bd218125b71b2",
        "torch_xcpu.ops.gumbel_sample",
        source_version=(
            "v0.31 separate cache source supported; upstream=0f8b398158f01fb3"
        ),
    ),
    _operator(
        "sample",
        "vllm.v1.worker.gpu.sample.gumbel",
        "_gumbel_sample_kernel",
        "6b671b965ff7dec1feeba9b385c4b31120216c0bdeee33b19ad1e5634554d90e",
        "71178816cc67bbb0cb907fb78b21a22c09f47157e10df2cbd73da766c735f28f",
        "torch_xcpu.ops.gumbel_sample",
        source_version=(
            "v0.31 separate cache source supported; upstream=0f8b398158f01fb3"
        ),
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
        "45cd33684859d7bba5a631131e1927096f5ddf0b2ac6c380b26e4244afd92e53",
        "e1708c1127650d697d217701d4853c8046545944e45160efeba1cc53c9316ac0",
        "torch_xcpu.ops.apply_top_k_top_p",
        source_version=(
            "v0.31 XCPU full filter; local=dd24619b0d47831e; "
            "upstream=bf95f58d10891293,aed894c190aa3db7,1ee7f78e806b5e9b"
        ),
    ),
    _operator(
        "grouped_topk",
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
        "grouped_topk",
        "3063007e0e777337365ac214ce1b17b872ff29ecbb31dea8c190b192cffa3999",
        "4f7fbcf35818ef44b6a629633d8d8145be49b28e7bdb7380e59600957800964f",
        "torch_xcpu.ops.grouped_topk",
        source_version=(
            "v0.31 grouped routing; sorted batch-invariant path remains native"
        ),
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
    _operator(
        "dflash",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "_prepare_dflash_inputs_kernel",
        "83672293d7f0a71c1d362afbe6fce746390c1f5b8cf62acd3f55717342f6e8a2",
        "3935dda267b0e226c55f35073f9cda77a9c42506b374aaf4f7ad58349b68e621",
        "torch_xcpu.ops.prepare_dflash_inputs",
        source_version="9c77da8aee (v0.31)",
    ),
    _operator(
        "dflash",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "prepare_dflash_inputs",
        "f82763cee5a906ce5c0b4fd152e8535d9b5fdf44be77a0fba31c81b6b53c9cf0",
        "2a88c1102efc8c4ae12d1e08853b3ee4de8b6219dbfab3dcdd2870c3abbe8d97",
        "torch_xcpu.ops.prepare_dflash_inputs",
        source_version="9c77da8aee (v0.31)",
    ),
    _operator(
        "dflash",
        "vllm.v1.worker.gpu.cp_utils",
        "cp_local_slot",
        "5919deceb1fb929a6cdcdddadb06e05d5535904316c9141801253e163236869c",
        "8ed0389159655f9a43af485ee78c4c0a011696b9a8f147509c7aa56860a64e56",
        "torch_xcpu.ops.prepare_dflash_inputs",
        source_version="9c77da8aee (v0.31)",
    ),
    _operator(
        "qwen3_dflash2",
        "vllm.model_executor.models.qwen3_dflash2",
        "_grouped_conv",
        "6072832ccff0d793b8dde8da270bb6d098533ca897f54fe830a32f75b458412b",
        "32a56e99e045e5a031b623a3083fb1cc2144730a77ce113156bc0045fd38276c",
        "torch_xcpu.ops.dflash2_grouped_conv",
        source_version="9c77da8aee (v0.31)",
    ),
    _operator(
        "qwen3_dflash2",
        "vllm.model_executor.models.qwen3_dflash2",
        "_score_edges",
        "f733826766f34777447b2de500c9fa74576f85afd6f3ad9c6be27e4175736106",
        "9eb583888e04087c87bb2a14ff69beb3933ca4a7de22a090b833a5d4e4fd6a2d",
        "torch_xcpu.ops.einsum_blpr_blcr_to_blpc",
        source_version="9c77da8aee (v0.31)",
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
    if actual_signature_hash not in (
        target.expected_signature_hash,
        *target.alternate_signature_hashes,
    ):
        expected = (target.expected_signature_hash, *target.alternate_signature_hashes)
        changed.append(
            f"signature hash expected one of {expected}, got {actual_signature_hash}"
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
    for source_file in UPSTREAM_SOURCE_FILES:
        if selected is not None and source_file.category not in selected:
            continue
        module = importlib.import_module(source_file.module)
        module_path = Path(inspect.getfile(module))
        actual_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
        if actual_sha256 != source_file.expected_sha256:
            raise KernelVersionError(
                f"{source_file.module}: upstream source file changed "
                f"(expected SHA256 {source_file.expected_sha256}, got "
                f"{actual_sha256}). Review all helper and metadata changes "
                "against the XCPU sparse-indexer translation before updating "
                "the file hash in upstream_compatibility.py."
            )
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
            or actual_signature_hash
            not in (target.expected_signature_hash, *target.alternate_signature_hashes)
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
