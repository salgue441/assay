from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from assay.rating import RATINGS, CriterionScore, Decision, Rating, rating_id
from assay.storage import ParquetStore, Partition


def make_rating(**changes: object) -> Rating:
    fields: dict[str, object] = {
        "rating_id": rating_id(document_id="d" * 64, rater_id="r1", rubric_hash="h" * 64),
        "document_id": "d" * 64,
        "unit_id": "owner/shapes:shapes/core.py#shapes.core",
        "repo": "owner/shapes",
        "commit": "c" * 40,
        "rater_id": "r1",
        "rubric_version": "v1",
        "rubric_hash": "e" * 64,
        "scores": (
            CriterionScore(criterion="accuracy", score=4),
            CriterionScore(criterion="clarity", score=2),
        ),
        "overall": 3,
        "decision": Decision.REVISE,
        "notes": "",
        "seconds_spent": 42.5,
        "rated_at": datetime(2026, 10, 19, tzinfo=UTC),
    }
    return Rating.model_validate(fields | changes)


def test_scores_are_looked_up_by_criterion() -> None:
    rating = make_rating()

    assert (rating.score_for("accuracy"), rating.score_for("clarity")) == (4, 2)


def test_round_trips_through_storage_with_nested_scores(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    rating = make_rating()

    store.write(RATINGS, [rating], Partition("owner/shapes", 0))

    assert store.read(RATINGS) == [rating]


@pytest.mark.parametrize("rater", ["Carlos", "carlos@example.com", "", "r 1", "x" * 33])
def test_rater_ids_must_be_short_lowercase_pseudonyms(rater: str) -> None:
    with pytest.raises(ValidationError):
        make_rating(rater_id=rater)


def test_each_criterion_is_scored_once() -> None:
    twice = (CriterionScore(criterion="accuracy", score=4),) * 2

    with pytest.raises(ValidationError, match="scored once"):
        make_rating(scores=twice)


@pytest.mark.parametrize("change", [{"overall": 0}, {"overall": 6}, {"scores": ()}])
def test_scores_must_be_on_the_scale(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_rating(**change)


def test_rating_id_depends_on_document_rater_and_rubric() -> None:
    base = {"document_id": "d" * 64, "rater_id": "r1", "rubric_hash": "h" * 64}

    assert rating_id(**base) == rating_id(**base)
    for key, value in (("document_id", "e" * 64), ("rater_id", "r2"), ("rubric_hash", "i" * 64)):
        assert rating_id(**(base | {key: value})) != rating_id(**base)
