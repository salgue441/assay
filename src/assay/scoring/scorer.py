"""
Score documents: weight their components into a composite and route them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from assay.generate import Document
from assay.judge import Judgment
from assay.metrics import MetricRow
from assay.scoring.components import critical_failures, judged_components, static_components
from assay.scoring.config import ScorerConfig
from assay.scoring.scores import Component, Score

_COMPOSITE_DIGITS = 9
"""Rounding that keeps floating-point noise from moving a composite across a
threshold, such as 79.99999999999999 for an exact 80."""


@dataclass
class ScoringReport:
    scores: list[Score] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)
    """Ids of documents with no weighted component to score them on."""


class CompositeScorer:
    """
    Combine each document's static metrics and one judge's grades into a
    0-100 composite and a routing decision.

    Only components with a positive weight count. When a document lacks some
    of them, the remaining weights are scaled up to sum to 1 and the missing
    names are recorded, so a document that could not be judged is still
    scored on its metrics. A critical metric failure rejects the document but
    its composite is still computed.
    """

    def __init__(self, config: ScorerConfig) -> None:
        self._config = config
        self._weights = {name: w for name, w in config.weights.items() if w > 0}

    def score(
        self,
        documents: Iterable[Document],
        metric_rows: Iterable[MetricRow],
        judgments: Iterable[Judgment],
    ) -> ScoringReport:
        """
        Score every document. `metric_rows` must hold one metric version
        and `judgments` one judging configuration per judge, or `ValueError`
        is raised. Rows for other documents are ignored.
        """

        rows = list(metric_rows)
        static = static_components(rows)
        judged = judged_components(judgments, judge=self._config.judge)
        critical = critical_failures(rows)

        report = ScoringReport()
        for document in documents:
            values = _merge(
                document.document_id,
                static.get(document.document_id, {}),
                judged.get(document.document_id, {}),
            )
            score = self._score(document, values, critical.get(document.document_id, ()))
            if score is None:
                report.unscored.append(document.document_id)
            else:
                report.scores.append(score)
        return report

    def _score(
        self, document: Document, values: Mapping[str, float], critical_reasons: tuple[str, ...]
    ) -> Score | None:
        present = sorted(name for name in self._weights if name in values)
        if not present:
            return None

        total_weight = sum(self._weights[name] for name in present)
        components = tuple(
            Component(name=name, value=values[name], weight=self._weights[name] / total_weight)
            for name in present
        )
        weighted = sum(c.value * c.weight for c in components)
        composite = min(100.0, round(100 * weighted, _COMPOSITE_DIGITS))
        critical = bool(critical_reasons)
        return Score(
            document_id=document.document_id,
            unit_id=document.unit_id,
            repo=document.repo,
            commit=document.commit,
            subject=document.subject,
            scorer_version=self._config.version,
            judge=self._config.judge,
            composite=composite,
            components=components,
            missing=tuple(sorted(name for name in self._weights if name not in values)),
            critical=critical,
            critical_reasons=critical_reasons,
            decision=self._config.route(composite, critical=critical),
        )


def _merge(
    document_id: str, static: Mapping[str, float], judged: Mapping[str, float]
) -> Mapping[str, float]:
    clashing = sorted(static.keys() & judged.keys())
    if clashing:
        msg = (
            f"document {document_id[:12]}: {', '.join(clashing)} is both a metric and a "
            "judged criterion, so its weight is ambiguous"
        )
        raise ValueError(msg)
    return {**static, **judged}
