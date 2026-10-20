"""
Measure stored documents: the `metrics` table.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import Field

from assay.corpus import CommitSha, SymbolRow, Unit
from assay.generate import Document
from assay.metrics.base import Metric
from assay.metrics.context import MetricContext
from assay.metrics.coverage import SymbolCoverage
from assay.metrics.integrity import ReferentialIntegrity
from assay.metrics.readability import Readability
from assay.metrics.rules import LanguageRules
from assay.metrics.structure import StructuralCompleteness
from assay.records import Record
from assay.storage import TableSpec


class MetricRow(Record):
    """One static metric for one document: a `metrics` row."""

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str
    repo: str
    commit: CommitSha
    subject: str
    metric: str
    metric_version: str
    value: float = Field(ge=0, le=1)
    critical: bool
    explanation: str


METRICS_TABLE = TableSpec(
    name="metrics", model=MetricRow, key=("document_id", "metric", "metric_version")
)


def static_metrics(*, integrity_critical_below: float | None = None) -> tuple[Metric, ...]:
    """The deterministic metrics, in the order they are reported."""

    return (
        ReferentialIntegrity(critical_below=integrity_critical_below),
        SymbolCoverage(),
        StructuralCompleteness(),
        Readability(),
    )


def measure(
    documents: Iterable[Document],
    units: Iterable[Unit],
    symbols: Iterable[SymbolRow],
    *,
    metrics: Sequence[Metric],
    rules: LanguageRules,
) -> list[MetricRow]:
    """
    Score every document with every metric.

    `units` and `symbols` must cover the documents' snapshots; a document
    whose unit is missing raises `ValueError`, since the tables disagree.
    """

    units_by_key = {(unit.unit_id, unit.commit): unit for unit in units}
    symbols_by_snapshot: dict[tuple[str, str], list[SymbolRow]] = defaultdict(list)
    for symbol in symbols:
        symbols_by_snapshot[(symbol.repo, symbol.commit)].append(symbol)

    rows: list[MetricRow] = []
    contexts: dict[tuple[str, str], MetricContext] = {}
    for document in documents:
        key = (document.unit_id, document.commit)
        unit = units_by_key.get(key)
        if unit is None:
            msg = f"document {document.document_id[:12]} cites unit {key} that is not stored"
            raise ValueError(msg)
        if key not in contexts:
            snapshot = symbols_by_snapshot[(unit.repo, unit.commit)]
            contexts[key] = MetricContext.build(unit, snapshot, rules)

        for metric in metrics:
            result = metric.score(document.text, contexts[key])
            rows.append(
                MetricRow(
                    document_id=document.document_id,
                    unit_id=document.unit_id,
                    repo=document.repo,
                    commit=document.commit,
                    subject=document.subject,
                    metric=metric.name,
                    metric_version=metric.version,
                    value=result.value,
                    critical=result.critical,
                    explanation=result.explanation,
                )
            )
    return rows
