"""
A rater's working session: what to rate next and how ratings are saved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from assay.corpus import SOURCES, UNITS, Unit
from assay.generate import DOCUMENTS, Document
from assay.judge import Rubric
from assay.rating.assignment import assigned
from assay.rating.ratings import (
    RATER_ID_PATTERN,
    RATINGS,
    CriterionScore,
    Decision,
    Rating,
    rating_id,
)
from assay.storage import ParquetStore, Partition


@dataclass(frozen=True)
class RatingItem:
    """One document to rate, with the source it documents. Nothing that
    identifies the model that wrote it is shown to the rater."""

    document: Document
    unit: Unit
    language: str


class RatingSession:
    """
    Serves one rater the documents assigned to them, in their order, and
    saves each rating to the partition of the document it rates.

    Documents the rater has already rated under this rubric are skipped, so
    a session can be closed and resumed at any point.
    """

    def __init__(
        self,
        store: ParquetStore,
        rubric: Rubric,
        *,
        rater_id: str,
        seed: int,
        share: float = 1.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not re.fullmatch(RATER_ID_PATTERN, rater_id):
            msg = f"rater id {rater_id!r} must be a short lowercase pseudonym such as 'r1'"
            raise ValueError(msg)
        self._store = store
        self._rubric = rubric
        self._rater_id = rater_id
        self._clock = clock

        snapshots = {(s.repo, s.commit): s for s in store.read(SOURCES)}
        units = {(u.unit_id, u.commit): u for u in store.read(UNITS)}
        self._items = [
            RatingItem(
                document=document,
                unit=units[(document.unit_id, document.commit)],
                language=snapshots[(document.repo, document.commit)].language.value,
            )
            for document in assigned(
                store.read(DOCUMENTS), rater_id=rater_id, seed=seed, share=share
            )
        ]
        self._epochs = {key: snapshot.epoch for key, snapshot in snapshots.items()}
        self._rated = {
            r.document_id
            for r in store.read(RATINGS)
            if r.rater_id == rater_id and r.rubric_hash == rubric.hash
        }

    @property
    def rater_id(self) -> str:
        return self._rater_id

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def progress(self) -> tuple[int, int]:
        """(rated, assigned) for this rater."""

        done = sum(item.document.document_id in self._rated for item in self._items)
        return done, len(self._items)

    def next_item(self) -> RatingItem | None:
        return next(
            (item for item in self._items if item.document.document_id not in self._rated), None
        )

    def submit(
        self,
        item: RatingItem,
        *,
        scores: Mapping[str, int],
        overall: int,
        decision: Decision,
        notes: str = "",
        seconds_spent: float = 0.0,
    ) -> Rating:
        """
        Save a rating. Every rubric criterion needs a score; a rating with a
        missing or unknown criterion raises `ValueError` and is not saved.
        """

        expected, given = sorted(self._rubric.criteria), sorted(scores)
        if given != expected:
            msg = f"score every criterion exactly once: expected {expected}, got {given}"
            raise ValueError(msg)

        document = item.document
        rating = Rating(
            rating_id=rating_id(
                document_id=document.document_id,
                rater_id=self._rater_id,
                rubric_hash=self._rubric.hash,
            ),
            document_id=document.document_id,
            unit_id=document.unit_id,
            repo=document.repo,
            commit=document.commit,
            rater_id=self._rater_id,
            rubric_version=self._rubric.version,
            rubric_hash=self._rubric.hash,
            scores=tuple(
                CriterionScore(criterion=name, score=scores[name]) for name in self._rubric.criteria
            ),
            overall=overall,
            decision=decision,
            notes=notes.strip(),
            seconds_spent=seconds_spent,
            rated_at=self._clock(),
        )

        partition = Partition(document.repo, self._epochs[(document.repo, document.commit)])
        stored = self._store.read(RATINGS, repo=partition.repo, epoch=partition.epoch)
        kept = [r for r in stored if r.rating_id != rating.rating_id]
        self._store.write(RATINGS, [*kept, rating], partition)
        self._rated.add(document.document_id)
        return rating
