# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class KernelSource:
    """Manually reviewed provenance, independent of runtime fingerprints.

    local_commit pins the committed vLLM checkout when the replacement changed.
    main_commit records its upstream main baseline; a backport instead records
    the upstream change in backport_commit. Neither is inferred from the tag.
    """

    tag: str
    local_commit: str
    main_commit: str | None = None
    backport_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.tag.strip() or any(c in self.tag for c in ";\r\n"):
            raise ValueError("tag must be a non-empty manual label without separators")
        if (self.main_commit is None) == (self.backport_commit is None):
            raise ValueError(
                "exactly one of main_commit and backport_commit is required"
            )
        for name in ("local_commit", "main_commit", "backport_commit"):
            value = getattr(self, name)
            if value is None and name != "local_commit":
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
                raise ValueError(f"{name} must be a full 40-character commit ID")

    def __str__(self) -> str:
        origin = (
            f"main={self.main_commit}"
            if self.main_commit is not None
            else f"backport={self.backport_commit}"
        )
        return f"{self.tag}; local={self.local_commit}; {origin}"


@dataclass(frozen=True)
class LegacyKernelSource:
    """Preserve incomplete historical evidence; never use for new audits.

    An old label alone cannot establish a local snapshot or a main/backport
    relationship. Keep it visible instead of manufacturing commit IDs.
    """

    label: str

    def __str__(self) -> str:
        return f"legacy-unverified: {self.label}"
