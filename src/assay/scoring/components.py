"""
Turn stored metric rows and judgments into named component values in [0, 1].
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import median

from assay.judge import MAX_SCORE, MIN_SCORE, Judgment
from assay.metrics import MetricRow

ComponentValues = Mapping[str, Mapping[str, float]]
"""Component values by document id, then component name."""


def static_components(rows: Iterable[MetricRow]) -> ComponentValues:
    """
    Each metric's value per document, named after the metric.

    Raises `ValueError` if a document has the same metric twice, which means
    rows from several metric versions were passed together.
    """

    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_metric = values[row.document_id]
        if row.metric in by_metric:
            msg = (
                f"document {row.document_id[:12]} has metric {row.metric!r} more than once; "
                "pass the rows of one metric version"
            )
            raise ValueError(msg)
        by_metric[row.metric] = row.value
    return values


def critical_failures(rows: Iterable[MetricRow]) -> Mapping[str, tuple[str, ...]]:
    """The sorted names of the metrics that failed critically, per document."""

    failed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.critical:
            failed[row.document_id].add(row.metric)
    return {document_id: tuple(sorted(names)) for document_id, names in failed.items()}


def judged_components(judgments: Iterable[Judgment], *, judge: str) -> ComponentValues:
    """
    One judge's grade per document and criterion, named after the criterion:
    the median over that judge's runs, mapped from the rubric scale onto
    [0, 1]. Other judges' judgments are ignored.

    Raises `ValueError` if a run appears twice for the same document and
    criterion, which means judgments from several settings (such as two
    temperatures or rubric versions) were passed together.
    """

    runs: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for judgment in judgments:
        if judgment.judge != judge:
            continue
        by_run = runs[judgment.document_id][judgment.criterion]
        if judgment.run_index in by_run:
            msg = (
                f"document {judgment.document_id[:12]} has run {judgment.run_index} of "
                f"{judgment.criterion!r} by {judge!r} more than once; "
                "pass the judgments of one judging configuration"
            )
            raise ValueError(msg)
        by_run[judgment.run_index] = judgment.score

    return {
        document_id: {
            criterion: _unit_interval(median(scores.values()))
            for criterion, scores in by_criterion.items()
        }
        for document_id, by_criterion in runs.items()
    }


def _unit_interval(score: float) -> float:
    return (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
