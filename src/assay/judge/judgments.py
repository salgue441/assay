"""
Judgments of documents: the `judgments` table.
"""

from __future__ import annotations

import hashlib

from pydantic import AwareDatetime, Field

from assay.corpus import CommitSha
from assay.judge.rubric import MAX_SCORE, MIN_SCORE
from assay.llm import Provider
from assay.records import Record
from assay.storage import TableSpec


class Judgment(Record):
    """
    One judge's grade of one document on one criterion in one run: a
    `judgments` row.

    `judgment_id` covers everything that determines the grade (document,
    rubric, reply schema, criterion, exact judge weights, run, seed,
    temperature and output limit), so repeating a run is skipped and any
    change adds a new judgment.
    """

    judgment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str
    repo: str
    commit: CommitSha
    subject: str
    """The subject model that wrote the document."""
    judge: str
    """The configured name of the judge model, such as `gemma3-4b`."""
    provider: Provider
    model_id: str
    model_revision: str
    rubric_version: str
    rubric_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion: str
    run_index: int = Field(ge=0)
    seed: int
    temperature: float = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    reply_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE)
    issues: tuple[str, ...]
    rationale: str
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    judged_at: AwareDatetime

    @property
    def contradicts_rubric(self) -> bool:
        """A top score despite listed problems, which the rubric forbids."""

        return self.score == MAX_SCORE and bool(self.issues)


JUDGMENTS = TableSpec(name="judgments", model=Judgment, key=("judgment_id",))


def judgment_id(
    *,
    document_id: str,
    rubric_hash: str,
    criterion: str,
    provider: Provider,
    model_id: str,
    model_revision: str,
    run_index: int,
    seed: int,
    temperature: float,
    max_output_tokens: int,
    reply_schema_hash: str,
) -> str:
    parts = (
        document_id,
        rubric_hash,
        reply_schema_hash,
        criterion,
        provider.value,
        model_id,
        model_revision,
        str(run_index),
        str(seed),
        repr(temperature),
        str(max_output_tokens),
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
