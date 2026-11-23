# SPDX-License-Identifier: Apache-2.0
"""Lightweight container for SP-sharded DSpark auxiliary hidden states.

Kept separate from ``model.py`` so generic DSpark registration can import the
type without triggering ``install_deepseek_v41()``.
"""

from typing import NamedTuple

import torch


class SpShardedAuxHiddenStates(NamedTuple):
    """Packed SP-sharded target aux hidden states handed to DSpark."""

    packed_shard: torch.Tensor
    full_num_tokens: int
