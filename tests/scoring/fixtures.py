"""Hand-built documents, metric rows and judgments for the scoring tests."""

import hashlib
from datetime import UTC, datetime

from assay.extract import SymbolKind
from assay.generate import Document
from assay.judge import Judgment
from assay.llm import Provider
from assay.metrics import MetricRow

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
COMMIT = "c" * 40
HASH = "0" * 64


def doc_id(name: str) -> str:
    """A stable 64-hex document id for a short name such as `a`."""

    return hashlib.sha256(name.encode()).hexdigest()


def document(name: str) -> Document:
    return Document(
        document_id=doc_id(name),
        unit_id=f"pkg.{name}",
        unit_hash=HASH,
        repo="owner/repo",
        commit=COMMIT,
        qualified_name=f"pkg.{name}",
        kind=SymbolKind.FUNCTION,
        subject="writer",
        provider=Provider.OLLAMA,
        model_id="writer:1b",
        model_revision="r1",
        prompt_version="v1",
        prompt_hash=HASH,
        seed=1,
        temperature=0.0,
        max_output_tokens=64,
        text="## doc\nText.",
        truncated=False,
        prompt_tokens=10,
        output_tokens=5,
        latency_seconds=0.1,
        cost_usd=0.0,
        generated_at=NOW,
    )


def metric_row(
    name: str, metric: str, value: float, *, critical: bool = False, version: str = "1"
) -> MetricRow:
    return MetricRow(
        document_id=doc_id(name),
        unit_id=f"pkg.{name}",
        repo="owner/repo",
        commit=COMMIT,
        subject="writer",
        metric=metric,
        metric_version=version,
        value=value,
        critical=critical,
        explanation="",
    )


def judgment(
    name: str, criterion: str, score: int, *, run: int = 0, judge: str = "grader"
) -> Judgment:
    return Judgment(
        judgment_id=hashlib.sha256(f"{name}|{criterion}|{run}|{judge}".encode()).hexdigest(),
        document_id=doc_id(name),
        unit_id=f"pkg.{name}",
        repo="owner/repo",
        commit=COMMIT,
        subject="writer",
        judge=judge,
        provider=Provider.OLLAMA,
        model_id=f"{judge}:1b",
        model_revision="r1",
        rubric_version="v1",
        rubric_hash=HASH,
        criterion=criterion,
        run_index=run,
        seed=100 + run,
        temperature=0.0,
        max_output_tokens=256,
        reply_schema_hash=HASH,
        score=score,
        issues=(),
        rationale="fits level",
        prompt_tokens=10,
        output_tokens=5,
        latency_seconds=0.1,
        cost_usd=0.0,
        judged_at=NOW,
    )
