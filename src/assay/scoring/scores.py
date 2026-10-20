"""
Composite scores of documents and their routing decisions: the `scores` table.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assay.corpus import CommitSha
from assay.records import Record
from assay.storage import TableSpec


class RoutingDecision(StrEnum):
    """What happens to a scored document next, as defined in the SAD."""

    PUBLISH = "publish"
    REVISE = "revise"
    ESCALATE = "escalate"
    REJECT = "reject"
    """A critical failure, whatever the composite."""


class Component(BaseModel):
    """One input to a composite: a static metric or a judged criterion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    """The weight actually applied, after renormalising over the components
    the document has."""


class Score(Record):
    """
    One document's composite under one scorer configuration: a `scores` row.

    A rejected document keeps its composite: rejection is a decision, not a
    score of 0, so calibration can still compare it with human ratings.
    """

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str
    repo: str
    commit: CommitSha
    subject: str
    scorer_version: str = Field(pattern=r"^[0-9a-f]{16}$")
    judge: str
    """The judge whose grades supplied the judged components."""
    composite: float = Field(ge=0, le=100)
    components: tuple[Component, ...] = Field(min_length=1)
    missing: tuple[str, ...]
    """Weighted components the document has no value for, such as judged
    criteria for a document too large to judge."""
    critical: bool
    critical_reasons: tuple[str, ...]
    """The metrics that reported a critical failure, sorted."""
    decision: RoutingDecision

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.critical != bool(self.critical_reasons):
            msg = "critical must be set exactly when there are critical reasons"
            raise ValueError(msg)
        if self.critical != (self.decision is RoutingDecision.REJECT):
            msg = "a document is rejected exactly when it has a critical failure"
            raise ValueError(msg)
        return self


SCORES = TableSpec(name="scores", model=Score, key=("document_id", "scorer_version"))
