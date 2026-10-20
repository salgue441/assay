"""
Judge variance: how much the judge's grades move between repeated runs of
the same document, and whether that movement changes the routing decision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev

from assay.calibrate.ablation import composite_of
from assay.calibrate.decisions import Routing
from assay.judge import MAX_SCORE, MIN_SCORE, Judgment
from assay.scoring import RoutingDecision, Score

UNSTABLE_SHOWN = 10


@dataclass(frozen=True)
class CriterionSpread:
    criterion: str
    mean_sd: float
    """Mean over documents of the standard deviation of their grades."""
    max_sd: float
    changed: float
    """Share of documents whose grade differs between at least two runs."""
    documents: int


@dataclass(frozen=True)
class UnstableDocument:
    document_id: str
    unit_id: str
    decisions: tuple[RoutingDecision, ...]
    """The routing decision under each run, in run order."""


@dataclass(frozen=True)
class VarianceStudy:
    """Repeated judging at one temperature."""

    temperature: float
    runs: int
    """Most runs any document has."""
    criteria: tuple[CriterionSpread, ...]
    band_stability: float | None
    """Share of documents whose routing decision is the same under every
    run; None when no document has two complete runs."""
    documents: int
    """Documents with at least two complete runs, behind `band_stability`."""
    unstable: tuple[UnstableDocument, ...]


def judge_variance(
    scores: Iterable[Score],
    judgments: Iterable[Judgment],
    *,
    weights: Mapping[str, float],
    criteria: Sequence[str],
    routing: Routing,
) -> list[VarianceStudy]:
    """
    One study per judging temperature. For each run, a document's composite
    is formed from its static metrics (from its score) and that run's grades
    alone, then routed; a document is stable when every run routes it the
    same way. Only documents in `scores` count, and only runs that graded
    every one of `criteria`.
    """

    by_document = {s.document_id: s for s in scores}
    grades: dict[float, dict[str, dict[int, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for judgment in judgments:
        if judgment.document_id in by_document and judgment.criterion in criteria:
            runs = grades[judgment.temperature][judgment.document_id]
            runs[judgment.run_index][judgment.criterion] = judgment.score

    return [
        _study(
            temperature,
            documents,
            scores=by_document,
            weights=weights,
            criteria=criteria,
            routing=routing,
        )
        for temperature, documents in sorted(grades.items())
    ]


def _study(
    temperature: float,
    documents: Mapping[str, Mapping[int, Mapping[str, int]]],
    *,
    scores: Mapping[str, Score],
    weights: Mapping[str, float],
    criteria: Sequence[str],
    routing: Routing,
) -> VarianceStudy:
    spreads = []
    for criterion in criteria:
        per_document = [
            [run[criterion] for run in runs.values() if criterion in run]
            for runs in documents.values()
        ]
        repeated = [g for g in per_document if len(g) >= 2]
        if not repeated:
            continue
        sds = [pstdev(g) for g in repeated]
        spreads.append(
            CriterionSpread(
                criterion=criterion,
                mean_sd=mean(sds),
                max_sd=max(sds),
                changed=sum(len(set(g)) > 1 for g in repeated) / len(repeated),
                documents=len(repeated),
            )
        )

    routed: list[tuple[str, tuple[RoutingDecision, ...]]] = []
    for document_id, runs in sorted(documents.items()):
        complete = [run for _, run in sorted(runs.items()) if set(criteria) <= set(run)]
        if len(complete) < 2:
            continue
        score = scores[document_id]
        static = {c.name: c.value for c in score.components if c.name not in criteria}
        decisions = []
        for run in complete:
            values = {**static, **{c: _unit(g) for c, g in run.items() if c in criteria}}
            value = composite_of(values, weights)
            if value is not None:
                decisions.append(routing.route(value, critical=score.critical))
        if len(decisions) >= 2:
            routed.append((document_id, tuple(decisions)))

    unstable = [
        UnstableDocument(document_id=d, unit_id=scores[d].unit_id, decisions=decisions)
        for d, decisions in routed
        if len(set(decisions)) > 1
    ]
    return VarianceStudy(
        temperature=temperature,
        runs=max(len(runs) for runs in documents.values()),
        criteria=tuple(spreads),
        band_stability=(1 - len(unstable) / len(routed)) if routed else None,
        documents=len(routed),
        unstable=tuple(unstable[:UNSTABLE_SHOWN]),
    )


def _unit(grade: int) -> float:
    return (grade - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
