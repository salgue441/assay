from datetime import UTC, datetime
from pathlib import Path

import pytest

from assay.judge import Rubric
from assay.rating import RATINGS, Decision, RatingSession, assigned
from assay.storage import ParquetStore

from .fixtures import PROJECT_ROOT, make_store

RUBRIC = Rubric.load(PROJECT_ROOT / "config" / "rubrics", "v1")
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
ALL_CRITERIA = {"accuracy": 4, "clarity": 3, "consistency": 5}


def session(store: ParquetStore, rater: str = "r1", share: float = 1.0) -> RatingSession:
    return RatingSession(store, RUBRIC, rater_id=rater, seed=7, share=share, clock=lambda: NOW)


def rate(rating_session: RatingSession, **changes: object) -> None:
    item = rating_session.next_item()
    assert item is not None
    options: dict[str, object] = {
        "scores": ALL_CRITERIA,
        "overall": 4,
        "decision": Decision.PUBLISH,
    } | changes
    rating_session.submit(item, **options)  # type: ignore[arg-type]


def test_serves_documents_in_the_raters_order(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)

    item = session(store).next_item()

    assert item is not None
    assert item.document == assigned(documents, rater_id="r1", seed=7)[0]
    assert item.unit.unit_id == item.document.unit_id
    assert item.language == "python"


def test_submitting_saves_a_complete_rating_and_moves_on(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)
    rating_session = session(store)
    first = rating_session.next_item()
    assert first is not None

    rating = rating_session.submit(
        first,
        scores=ALL_CRITERIA,
        overall=3,
        decision=Decision.REVISE,
        notes="  misses the default  ",
        seconds_spent=61.0,
    )

    assert store.read(RATINGS) == [rating]
    assert [(s.criterion, s.score) for s in rating.scores] == list(ALL_CRITERIA.items())
    assert (rating.rater_id, rating.notes, rating.seconds_spent, rating.rated_at) == (
        "r1",
        "misses the default",
        61.0,
        NOW,
    )
    assert rating_session.progress == (1, len(documents))
    assert rating_session.next_item() != first


def test_a_new_session_resumes_where_the_rater_stopped(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)
    first_session = session(store)
    rate(first_session)
    rate(first_session)

    resumed = session(store)

    assert resumed.progress == (2, len(documents))
    assert resumed.next_item() == first_session.next_item()


def test_raters_are_independent(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)
    rate(session(store, "r1"))

    assert session(store, "r2").progress == (0, len(documents))


def test_rating_the_same_document_again_replaces_the_rating(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    rating_session = session(store)
    item = rating_session.next_item()
    assert item is not None

    rating_session.submit(item, scores=ALL_CRITERIA, overall=2, decision=Decision.ESCALATE)
    rating_session.submit(item, scores=ALL_CRITERIA, overall=5, decision=Decision.PUBLISH)

    (stored,) = store.read(RATINGS)
    assert (stored.overall, stored.decision) == (5, Decision.PUBLISH)


def test_finishing_every_document_leaves_nothing_next(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)
    rating_session = session(store)
    for _ in documents:
        rate(rating_session)

    assert rating_session.next_item() is None
    assert rating_session.progress == (len(documents), len(documents))


def test_an_additional_rater_gets_the_overlap_only(tmp_path: Path) -> None:
    store, documents = make_store(tmp_path)

    assert 0 < session(store, "r2", share=0.5).progress[1] < len(documents)


@pytest.mark.parametrize(
    "scores",
    [{"accuracy": 4, "clarity": 3}, ALL_CRITERIA | {"charm": 5}],
    ids=["missing", "unknown"],
)
def test_every_rubric_criterion_must_be_scored(tmp_path: Path, scores: dict[str, int]) -> None:
    store, _ = make_store(tmp_path)

    with pytest.raises(ValueError, match="score every criterion exactly once"):
        rate(session(store), scores=scores)
    assert store.read(RATINGS) == []


def test_rater_ids_must_be_pseudonyms(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)

    with pytest.raises(ValueError, match="lowercase pseudonym"):
        session(store, "Carlos Salguero")
