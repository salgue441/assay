"""
Ablation: how much each part of the scorer contributes to its agreement
with the raters, measured by removing that part and scoring again.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from assay.rating import Decision
from assay.scoring import RoutingDecision

if TYPE_CHECKING:
    from assay.calibrate.study import RatedDocument

JUDGE = "judge"
"""Removes every judged criterion at once."""
REJECT_RULE = "reject_rule"
"""Keeps rejected documents and routes them by their composite."""


@dataclass(frozen=True)
class Routing:
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
class Outcome:
    """Agreement of one scorer variant with the raters."""

    spearman: float | None
    """Composite against human overall over the documents the variant
    correlates; None when undefined, e.g. for constant scores."""
    decision_agreement: float | None
    """Share of documents with an undisputed human decision where the
    variant's decision matches it, counting `reject` as `escalate`."""
    documents: int
    """Documents the variant could score."""


@dataclass(frozen=True)
class Ablation:
    removed: str
    """A component name, `judge` or `reject_rule`."""
    outcome: Outcome
    delta_spearman: float | None
    """Ablated minus full; negative when the removed part helped."""
    delta_decision_agreement: float | None


@dataclass(frozen=True)
class AblationStudy:
    full: Outcome
    ablations: tuple[Ablation, ...]


def ablate(
    documents: Sequence[RatedDocument],
    *,
    weights: Mapping[str, float],
    judged: Collection[str],
    routing: Routing,
) -> AblationStudy:
    """
    Score `documents` with all of `weights`, then without each positively
    weighted component, without the `judged` criteria together, and without
    the reject rule. Weights are renormalised over what remains, as the
    scorer does for a document missing a component.
    """

    active = {name: w for name, w in weights.items() if w > 0}
    full = _outcome(documents, active, routing, reject_rule=True)

    variants: list[tuple[str, Mapping[str, float], bool]] = [
        (name, {n: w for n, w in active.items() if n != name}, True) for name in sorted(active)
    ]
    if judged_active := [n for n in active if n in judged]:
        variants.append((JUDGE, {n: w for n, w in active.items() if n not in judged_active}, True))
    variants.append((REJECT_RULE, active, False))

    ablations = []
    for removed, kept, reject_rule in variants:
        outcome = _outcome(documents, kept, routing, reject_rule=reject_rule)
        ablations.append(
            Ablation(
                removed=removed,
                outcome=outcome,
                delta_spearman=_delta(outcome.spearman, full.spearman),
                delta_decision_agreement=_delta(
                    outcome.decision_agreement, full.decision_agreement
                ),
            )
        )
    return AblationStudy(full=full, ablations=tuple(ablations))


def composite(document: RatedDocument, weights: Mapping[str, float]) -> float | None:
    """
    The 0-100 composite under `weights`, renormalised over the components
    the document has; None when it has none of them.
    """

    present = {n: w for n, w in weights.items() if w > 0 and n in document.components}
    total = sum(present.values())
    if total == 0:
        return None
    return 100 * sum(document.components[n] * w for n, w in present.items()) / total


def _outcome(
    documents: Sequence[RatedDocument],
    weights: Mapping[str, float],
    routing: Routing,
    *,
    reject_rule: bool,
) -> Outcome:
    scored = [(d, c) for d in documents if (c := composite(d, weights)) is not None]
    correlated = [(d, c) for d, c in scored if not (reject_rule and d.rejected)]
    decided = [
        (routing.route(c, critical=reject_rule and d.rejected), d.human_decision)
        for d, c in scored
        if d.human_decision is not None
    ]
    return Outcome(
        spearman=_spearman([c for _, c in correlated], [d.overall for d, _ in correlated]),
        decision_agreement=(
            sum(_matches(scorer, human) for scorer, human in decided) / len(decided)
            if decided
            else None
        ),
        documents=len(scored),
    )


def _matches(scorer: RoutingDecision, human: Decision | None) -> bool:
    expected = RoutingDecision.ESCALATE if scorer is RoutingDecision.REJECT else scorer
    return human is not None and expected.value == human.value


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(stats.spearmanr(x, y).statistic)


def _delta(value: float | None, reference: float | None) -> float | None:
    return None if value is None or reference is None else value - reference
