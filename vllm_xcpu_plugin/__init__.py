# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from vllm.logger import logger

# def xcpu_platform_plugin() -> Optional[str]:
#     return "vllm_xcpu_plugin.xcpu_platform_plugin.XcpuPlatform"


def register_ops():
    logger.info("register_ops")
    import vllm_xcpu_plugin.custom_ops  # noqa
