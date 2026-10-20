"""
The contract every static metric implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assay.metrics.context import MetricContext


@dataclass(frozen=True, slots=True)
class MetricResult:
    value: float
    """Higher is better, in [0, 1]."""
    explanation: str
    critical: bool = False
    """A defect that rejects the document whatever its composite score."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            msg = f"metric value must be in [0, 1], got {self.value}"
            raise ValueError(msg)


class Metric(Protocol):
    """
    A deterministic score for one document. `version` changes whenever the
    definition does, so stored values from different definitions never mix.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def score(self, text: str, context: MetricContext) -> MetricResult: ...


def ratio(part: int, whole: int) -> float:
    """`part / whole`, or 1.0 when there is nothing to measure."""

    return 1.0 if whole == 0 else part / whole


def listed(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(f"`{name}`" for name in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")
