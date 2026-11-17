# SPDX-License-Identifier: Apache-2.0
"""Fused CPU/MCPU speculative rejection sampling registration."""

from vllm.platforms import current_platform


def register_rejection_sampler() -> None:
    if current_platform.device_name != "mcpu":
        return
    from .patch import install_rejection_sampler

    install_rejection_sampler()


__all__ = ["register_rejection_sampler"]
