"""
Human ratings of documents: the `ratings` table.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from assay.corpus import CommitSha
from assay.judge import MAX_SCORE, MIN_SCORE
from assay.records import Record
from assay.storage import TableSpec

RATER_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"
"""A short pseudonym such as `r1`: lowercase, so names and emails stand out."""


class Decision(StrEnum):
    """What a rater would do with the document, as defined in the SAD."""

    PUBLISH = "publish"
    """Usable as is; no factual errors."""
    REVISE = "revise"
    """Factual, coverage or clarity problems a reviewer could fix by editing."""
    ESCALATE = "escalate"
    """Misleading or wrong in a way that needs the source re-read or the
    document regenerated."""


class CriterionScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: str = Field(min_length=1)
    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE)


class Rating(Record):
    """
    One rater's assessment of one document: a `ratings` row.

    `rating_id` covers the document, the rater and the rubric, so rating the
    same document again under the same rubric replaces the earlier rating.
    """

    rating_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str
    repo: str
    commit: CommitSha
    rater_id: str = Field(pattern=RATER_ID_PATTERN)
    rubric_version: str
    rubric_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[CriterionScore, ...] = Field(min_length=1)
    overall: int = Field(ge=MIN_SCORE, le=MAX_SCORE)
    decision: Decision
    notes: str
    seconds_spent: float = Field(ge=0)
    """Time from showing the document to submitting, for labeling cost."""
    rated_at: AwareDatetime

    @model_validator(mode="after")
    def _check_unique_criteria(self) -> Self:
        criteria = [s.criterion for s in self.scores]
        if len(set(criteria)) != len(criteria):
            msg = f"each criterion is scored once: {', '.join(criteria)}"
            raise ValueError(msg)
        return self

    def score_for(self, criterion: str) -> int:
        return next(s.score for s in self.scores if s.criterion == criterion)


RATINGS = TableSpec(name="ratings", model=Rating, key=("rating_id",))


def rating_id(*, document_id: str, rater_id: str, rubric_hash: str) -> str:
    parts = (document_id, rater_id, rubric_hash)
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
