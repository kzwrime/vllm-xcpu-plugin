# SPDX-License-Identifier: Apache-2.0
"""XCPU registration for generic DSpark speculator optimizations."""

from vllm.platforms import current_platform


def register_dspark() -> None:
    if current_platform.device_name != "mcpu":
        return
    from .speculator import install_dspark

    install_dspark()
