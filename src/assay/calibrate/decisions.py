"""
Routing decisions against the raters' decisions: how often they match, how
often an automatic publish is one the raters would publish, and which
thresholds make them match best.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from assay.rating import Decision
from assay.scoring import RoutingDecision

if TYPE_CHECKING:
    from assay.calibrate.study import RatedDocument

THRESHOLD_STEP = 1.0
"""Spacing of the candidate thresholds on the 0-100 composite."""


@dataclass(frozen=True)
class Routing:
    """Routing thresholds on the 0-100 composite, as the scorer applies them."""

    publish_at_or_above: float
    escalate_below: float

    def route(self, composite: float, *, critical: bool) -> RoutingDecision:
        if critical:
            return RoutingDecision.REJECT
        if composite >= self.publish_at_or_above:
            return RoutingDecision.PUBLISH
        if composite < self.escalate_below:
            return RoutingDecision.ESCALATE
        return RoutingDecision.REVISE


@dataclass(frozen=True)
class DecisionQuality:
    """Agreement of routing decisions with undisputed human decisions."""

    agreement: float | None
    """Share of decisions that match, counting `reject` as `escalate`."""
    publish_precision: float | None
    """Share of automatic publishes the raters would also publish; None when
    nothing was routed to publish."""
    published: int
    decided: int
    """Documents with an undisputed human decision."""


def matches(scorer: RoutingDecision, human: Decision) -> bool:
    """Whether a routing decision is the one the raters made; a rejection
    stands for escalation."""

    expected = RoutingDecision.ESCALATE if scorer is RoutingDecision.REJECT else scorer
    return expected.value == human.value


def decision_quality(pairs: Iterable[tuple[RoutingDecision, Decision | None]]) -> DecisionQuality:
    """Quality over (routing decision, human decision) pairs; pairs whose
    raters disagreed (None) are left out."""

    decided = [(scorer, human) for scorer, human in pairs if human is not None]
    published = [human for scorer, human in decided if scorer is RoutingDecision.PUBLISH]
    return DecisionQuality(
        agreement=(
            sum(matches(scorer, human) for scorer, human in decided) / len(decided)
            if decided
            else None
        ),
        publish_precision=(
            sum(human is Decision.PUBLISH for human in published) / len(published)
            if published
            else None
        ),
        published=len(published),
        decided=len(decided),
    )


def routed(documents: Sequence[RatedDocument], routing: Routing) -> DecisionQuality:
    """Decision quality of `documents` routed by `routing`, keeping the
    reject rule."""

    return decision_quality(
        (routing.route(d.composite, critical=d.rejected), d.human_decision) for d in documents
    )


def choose_routing(
    documents: Sequence[RatedDocument], *, step: float = THRESHOLD_STEP
) -> Routing | None:
    """
    The thresholds, on a grid of `step`, that make the most decisions on
    `documents` match the raters'. Ties go to the higher publish precision,
    then to the higher publish threshold and the higher escalate threshold,
    which publish less and escalate more. None when no document has an
    undisputed human decision.
    """

    if not any(d.human_decision is not None for d in documents):
        return None
    grid = [i * step for i in range(int(100 / step) + 1)]
    best: tuple[float, float, float, float] | None = None
    for publish in grid:
        for escalate in (t for t in grid if t <= publish):
            quality = routed(documents, Routing(publish, escalate))
            key = (quality.agreement or 0.0, quality.publish_precision or 0.0, publish, escalate)
            if best is None or key > best:
                best = key
    assert best is not None
    return Routing(publish_at_or_above=best[2], escalate_below=best[3])
