import hashlib
from datetime import UTC, datetime

import pytest

from assay.corpus import SymbolRow
from assay.generate import Document
from assay.llm import Provider
from assay.metrics import (
    METRICS_TABLE,
    MetricResult,
    MetricRow,
    get_rules,
    measure,
    static_metrics,
)

from .fixtures import SYMBOLS, UNITS


def document(unit_name: str, text: str, subject: str = "writer") -> Document:
    unit = UNITS[unit_name]
    return Document(
        document_id=hashlib.sha256(f"{unit.unit_id}|{subject}".encode()).hexdigest(),
        unit_id=unit.unit_id,
        unit_hash=unit.unit_hash,
        repo=unit.repo,
        commit=unit.commit,
        qualified_name=unit.qualified_name,
        kind=unit.kind,
        subject=subject,
        provider=Provider.OLLAMA,
        model_id="m",
        model_revision="r",
        prompt_version="v1",
        prompt_hash="0" * 64,
        seed=1,
        temperature=0.0,
        max_output_tokens=10,
        text=text,
        truncated=False,
        prompt_tokens=1,
        output_tokens=1,
        latency_seconds=0.0,
        cost_usd=0.0,
        generated_at=datetime(2026, 10, 19, tzinfo=UTC),
    )


def symbol_rows() -> list[SymbolRow]:
    return [
        SymbolRow(
            **s.model_dump(),
            repo="owner/textkit",
            commit="c" * 40,
            path="textkit/core.py",
            module="textkit.core",
            language="python",
            file_has_syntax_errors=False,
        )
        for s in SYMBOLS
    ]


def run(*documents: Document) -> list[MetricRow]:
    return measure(
        documents,
        UNITS.values(),
        symbol_rows(),
        metrics=static_metrics(integrity_critical_below=0.9),
        rules=get_rules("python"),
    )


def test_every_document_gets_every_metric() -> None:
    rows = run(document("wrap_text", "## `wrap_text`"), document("log", "## `log`", "other"))

    assert [(r.subject, r.metric) for r in rows] == [
        (subject, metric)
        for subject in ("writer", "other")
        for metric in (
            "referential_integrity",
            "symbol_coverage",
            "structural_completeness",
            "readability",
        )
    ]
    METRICS_TABLE.validate(METRICS_TABLE.to_arrow(rows))


def test_rows_carry_identity_values_and_critical_flags() -> None:
    doc = document("wrap_text", "`invented_name` only")

    integrity = run(doc)[0]

    assert (integrity.document_id, integrity.unit_id, integrity.commit) == (
        doc.document_id,
        doc.unit_id,
        doc.commit,
    )
    assert (integrity.metric_version, integrity.value, integrity.critical) == ("v1", 0.0, True)


def test_a_document_whose_unit_is_not_stored_is_an_error() -> None:
    orphan = document("wrap_text", "x").model_copy(update={"commit": "d" * 40})

    with pytest.raises(ValueError, match="that is not stored"):
        run(orphan)


def test_metric_results_must_be_in_range() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        MetricResult(value=1.5, explanation="too good")
