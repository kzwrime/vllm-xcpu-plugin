# SPDX-License-Identifier: Apache-2.0
"""Upstream contracts of the XCPU Conv2dLayer / Conv3dLayer OOT layers."""

from vllm_xcpu_plugin.fake_triton.provenance import KernelSource
from vllm_xcpu_plugin.upstream_contracts import (
    Manifest,
    Patch,
    UpstreamRef,
    WatchedModule,
)

_V031 = KernelSource(
    tag="v0.31.0",
    local_commit="0670c36056d45530524f498f731bf3847a7d4ce0",
    main_commit="283d76d56a45d9bbead5ba650402fb0957abc30c",
)

MANIFEST = Manifest(
    category="vision_conv",
    packages=(
        "vllm_xcpu_plugin.layers.conv",
        "vllm_xcpu_plugin.models.vision_conv",
    ),
    watched=(
        WatchedModule(
            module="vllm.model_executor.layers.conv",
            sha256="2def9222052fd0734467e531a61b6c2996ca87693bc9b352caad0f349b2bb159",
            source=_V031,
        ),
    ),
    patches=(
        Patch(
            id="vision_conv.conv2d",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.layers.conv:XcpuConv2dLayer.forward_oot",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.custom_op:CustomOp.forward_oot",
                    source_hash="2204d4feda3a77606c46395a0af2c62a4551430515d9c0eac5de700c288b99c3",
                    signature_hash="3037c3ba3a932c5f37a4b4d7a744f12737b140c3a1b9a5db95bf35362400046f",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:ConvLayerBase.__init__",
                    source_hash="cf3a77f8a43e0ee574ee199b1f86720b07f62913a859e929ab43a91bf85f587b",
                    signature_hash="cabef53b8b0202a6bc4d1a1694675486928351d2851f09b42f9e91e6633fc13c",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv2dLayer.forward_native",
                    source_hash="3b8a28e1d312ffe28cd958510950c4720fcb05be81e7f9cb8da6e7478e109aed",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv2dLayer._forward_mulmat",
                    source_hash="b493956042885cdfae9ac063db313b24efa5c388e927e9043d2ca03fcddbbeb1",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv2dLayer._forward_conv",
                    source_hash="d21bf2eab92924201855b5dbe411ef342c9e233c36c4a218ee81917a617d4d85",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
            ),
            contract=(
                "When enable_linear holds (kernel_size == stride, zero padding, "
                "groups == 1), XcpuConv2dLayer.forward_oot mirrors "
                "Conv2dLayer._forward_mulmat: unfold spatial windows (K1, K2), "
                "permute (0, 2, 3, 1, 4, 5) to (-1, input_size), invoke "
                "torch_xcpu.ops.linear with weight.view(out_channels, input_size) "
                "and bias instead of F.linear (avoiding native aten::mm / "
                "aten::addmm), and reshape/permute to (B, out_channels, H // K1, "
                "W // K2). Non-linear conv configurations delegate to "
                "forward_native (_forward_conv). Re-check: enable_linear predicate, "
                "parameter layout/dtypes in ConvLayerBase.__init__, unfold/permute "
                "order, and output shape/layout."
            ),
            impl=("torch_xcpu/torch_xcpu/ops_defs/gemm.py::linear",),
            tests=("vllm-xcpu-plugin/tests/layers/test_conv_oot.py",),
        ),
        Patch(
            id="vision_conv.conv3d",
            source=_V031,
            downstream=(
                "vllm_xcpu_plugin.layers.conv:XcpuConv3dLayer.forward_oot",
            ),
            upstream=(
                UpstreamRef(
                    target="vllm.model_executor.custom_op:CustomOp.forward_oot",
                    source_hash="2204d4feda3a77606c46395a0af2c62a4551430515d9c0eac5de700c288b99c3",
                    signature_hash="3037c3ba3a932c5f37a4b4d7a744f12737b140c3a1b9a5db95bf35362400046f",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:ConvLayerBase.__init__",
                    source_hash="cf3a77f8a43e0ee574ee199b1f86720b07f62913a859e929ab43a91bf85f587b",
                    signature_hash="cabef53b8b0202a6bc4d1a1694675486928351d2851f09b42f9e91e6633fc13c",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv3dLayer.forward_native",
                    source_hash="78f9bc455f9528cdf5ebe6d312d86b62c981dab90ad0ddd6ca60031d8dcca11a",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv3dLayer._forward_mulmat",
                    source_hash="064c42c4a2061f6ff13b403f0cf3ba1ac32307d139e1f08fe28eb61438a48d1c",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
                UpstreamRef(
                    target="vllm.model_executor.layers.conv:Conv3dLayer._forward_conv",
                    source_hash="e823dabd036e7565bfa904b3a518095dec2fae784aac99dbfa4c4d096c94050e",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
                UpstreamRef(
                    target="vllm.model_executor.models.qwen3_vl:Qwen3_VisionPatchEmbed.forward",
                    source_hash="be5c5b47cd62b6e3219b0487c540b84fb2efd67a080f94983d380a7d483c6868",
                    signature_hash="2dd66a951aaa896a270323740f4a7cde70ba2afc42986079c202e96bf88d6799",
                ),
            ),
            contract=(
                "When enable_linear holds (kernel_size == stride, zero padding, "
                "groups == 1), XcpuConv3dLayer.forward_oot mirrors "
                "Conv3dLayer._forward_mulmat: unfold spatiotemporal windows "
                "(K1, K2, K3), permute (0, 2, 3, 4, 1, 5, 6, 7) to (-1, input_size), "
                "invoke torch_xcpu.ops.linear with weight.view(out_channels, "
                "input_size) and bias instead of F.linear, and reshape/permute to "
                "(B, out_channels, T // K1, H // K2, W // K3). The logical output "
                "view returned by torch_xcpu.ops.linear must remain view-compatible "
                "with Qwen3_VisionPatchEmbed.forward (.view(L, hidden_size)) under "
                "dynamic-L torch.compile. Non-linear conv configurations delegate "
                "to forward_native (_forward_conv). Re-check: enable_linear "
                "predicate, unfold/permute order, and patch-embed caller reshapes."
            ),
            impl=("torch_xcpu/torch_xcpu/ops_defs/gemm.py::linear",),
            tests=("vllm-xcpu-plugin/tests/layers/test_conv_oot.py",),
        ),
    ),
)
