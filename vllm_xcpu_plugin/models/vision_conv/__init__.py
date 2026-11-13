# SPDX-License-Identifier: Apache-2.0
"""Vision patch-embedding convolution OOT registration on XCPU."""

from vllm_xcpu_plugin.upstream_compatibility import verify_upstream_compatibility


def register_vision_conv() -> None:
    verify_upstream_compatibility(("vision_conv",))
    import vllm_xcpu_plugin.layers.conv  # noqa: F401
