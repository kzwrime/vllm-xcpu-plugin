# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import logger


def xcpu_platform_plugin() -> str | None:
    return "vllm_xcpu_plugin.platform.XcpuPlatform"


def register_ops():
    logger.info("register_ops")
    import vllm_xcpu_plugin.custom_ops  # noqa
