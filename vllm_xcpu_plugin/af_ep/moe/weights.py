"""F-rank routed-expert checkpoint ownership and loading."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol

import torch

_ROUTED_WEIGHT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<relative>.+)$"
)
_PER_EXPERT_WEIGHT = re.compile(r"^(?P<expert>\d+)\.(?P<tail>.+)$")


class ExpertWeightTarget(Protocol):
    layer_indices: tuple[int, ...]
    num_experts: int
    ep_size: int

    def local_expert_ids(self) -> tuple[int, ...]: ...

    def load_routed_weight(
        self,
        layer_idx: int,
        relative_name: str,
        weight: torch.Tensor,
    ) -> Collection[str]: ...


@dataclass(frozen=True)
class WeightLoadAudit:
    loaded_checkpoint_names: frozenset[str]
    loaded_parameter_names: frozenset[str]
    remote_checkpoint_names: frozenset[str]
    skipped_non_routed_weights: int

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_checkpoint_names)


class RoutedExpertsWeightLoader:
    """Filter one EP shard, then delegate tensor formats to vLLM."""

    def __init__(self, target: ExpertWeightTarget) -> None:
        self.target = target
        assert target.layer_indices and target.num_experts > 0
        assert target.num_experts % target.ep_size == 0
        local_ids = target.local_expert_ids()
        assert len(local_ids) == target.num_experts // target.ep_size
        self._layers = frozenset(target.layer_indices)
        self._local_experts = frozenset(local_ids)

    def load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> WeightLoadAudit:
        seen: set[str] = set()
        loaded: set[str] = set()
        loaded_parameters: set[str] = set()
        remote: set[str] = set()
        loaded_layers: set[int] = set()
        skipped = 0

        for raw_name, weight in weights:
            name = self._normalize_checkpoint_name(raw_name)
            if name in seen:
                raise ValueError(f"duplicate checkpoint weight {raw_name!r}")
            seen.add(name)

            routed = _ROUTED_WEIGHT.fullmatch(name)
            if routed is None:
                skipped += 1
                continue

            layer_idx = int(routed.group("layer"))
            if layer_idx not in self._layers:
                raise ValueError(
                    f"routed weight belongs to a non-MoE layer: {name!r}"
                )
            relative_name = routed.group("relative")
            per_expert = _PER_EXPERT_WEIGHT.match(relative_name)
            if per_expert is not None:
                expert_id = int(per_expert.group("expert"))
                if not 0 <= expert_id < self.target.num_experts:
                    raise ValueError(
                        f"routed weight has invalid expert index: {name!r}"
                    )
                if expert_id not in self._local_experts:
                    remote.add(name)
                    continue

            consumed = self.target.load_routed_weight(
                layer_idx, relative_name, weight
            )
            if not consumed:
                raise ValueError(f"unsupported routed-expert weight {name!r}")
            loaded.add(name)
            loaded_parameters.update(
                f"routed_experts.{layer_idx}.{parameter_name}"
                for parameter_name in consumed
            )
            loaded_layers.add(layer_idx)

        missing_layers = self._layers - loaded_layers
        if missing_layers:
            raise ValueError(
                "no local routed-expert weights were loaded for layers "
                f"{sorted(missing_layers)}"
            )
        return WeightLoadAudit(
            loaded_checkpoint_names=frozenset(loaded),
            loaded_parameter_names=frozenset(loaded_parameters),
            remote_checkpoint_names=frozenset(remote),
            skipped_non_routed_weights=skipped,
        )

    @staticmethod
    def _normalize_checkpoint_name(name: str) -> str:
        prefix = "model.language_model."
        if name.startswith(prefix):
            return "model." + name.removeprefix(prefix)
        prefix = "language_model.model."
        if name.startswith(prefix):
            return "model." + name.removeprefix(prefix)
        return name
