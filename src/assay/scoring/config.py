"""
How a composite is formed and routed, and the version that identifies it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from assay.scoring.scores import RoutingDecision

WEIGHT_TOLERANCE = 1e-9
"""How far the weights may sum from 1, for floating-point rounding."""

_VERSION_CHARS = 16


@dataclass(frozen=True)
class ScorerConfig:
    """
    Component weights, the judge whose grades are used, and the routing
    thresholds on the 0-100 composite.

    `weights` maps a metric or criterion name to a non-negative weight; the
    weights sum to 1. A copy is kept, so later changes to the caller's
    mapping cannot change a configuration whose version was already used.
    """

    weights: Mapping[str, float]
    judge: str
    publish_at_or_above: float
    escalate_below: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        self._check_weights()
        self._check_thresholds()
        if not self.judge:
            msg = "judge must name a configured judge"
            raise ValueError(msg)

    def _check_weights(self) -> None:
        if not self.weights:
            msg = "weights must name at least one component"
            raise ValueError(msg)
        negative = sorted(name for name, weight in self.weights.items() if weight < 0)
        if negative:
            msg = f"weights must not be negative: {', '.join(negative)}"
            raise ValueError(msg)
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            msg = f"weights must sum to 1, got {total:.6f}"
            raise ValueError(msg)

    def _check_thresholds(self) -> None:
        for name, value in (
            ("publish_at_or_above", self.publish_at_or_above),
            ("escalate_below", self.escalate_below),
        ):
            if not 0 <= value <= 100:
                msg = f"{name} must be within 0-100, got {value}"
                raise ValueError(msg)
        if self.escalate_below > self.publish_at_or_above:
            msg = "escalate_below must not exceed publish_at_or_above"
            raise ValueError(msg)

    @cached_property
    def version(self) -> str:
        """
        A short hash of everything that determines a score, so scores from
        different configurations are stored side by side and never mixed.
        """

        canonical = json.dumps(
            {
                "weights": {name: float(weight) for name, weight in self.weights.items()},
                "judge": self.judge,
                "publish_at_or_above": float(self.publish_at_or_above),
                "escalate_below": float(self.escalate_below),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_VERSION_CHARS]

    def route(self, composite: float, *, critical: bool) -> RoutingDecision:
        """The decision for a document with this composite."""

        if critical:
            return RoutingDecision.REJECT
        if composite >= self.publish_at_or_above:
            return RoutingDecision.PUBLISH
        if composite < self.escalate_below:
            return RoutingDecision.ESCALATE
        return RoutingDecision.REVISE
