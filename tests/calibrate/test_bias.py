import json

import pytest

from assay.calibrate.bias import (
    MIN_CONTROLLED,
    BiasStudy,
    JudgedDocument,
    human_grades,
    judge_bias,
    judged_documents,
)
from assay.calibrate.report import bias_json, bias_text
from assay.judge import Judgment

from ..scoring.fixtures import document, judgment
from .test_study import RUBRIC, doc_id, rating

JUDGES = {"own": "phi", "other": "gemma"}
SUBJECTS = {"phi-writer": "phi", "qwen-writer": "qwen"}


def judged(
    name: str,
    *,
    unit: str | None = None,
    subject: str = "qwen-writer",
    words: int = 10,
    grades: dict[str, float] | None = None,
    human: float | None = None,
) -> JudgedDocument:
    """`grades` holds each judge's accuracy grade."""

    return JudgedDocument(
        document_id=doc_id(name),
        unit_id=unit or name,
        subject=subject,
        words=words,
        grades={judge: {"accuracy": grade} for judge, grade in (grades or {}).items()},
        human={} if human is None else {"accuracy": human},
    )


def study(documents: list[JudgedDocument], judges: dict[str, str] = JUDGES) -> BiasStudy:
    return judge_bias(
        documents,
        judges=judges,
        subjects=SUBJECTS,
        criteria=("accuracy",),
        resamples=200,
        seed=1,
    )


def pairs(gaps: list[tuple[float, float]]) -> list[JudgedDocument]:
    """One unit per entry, documented by both families; each entry holds the
    own-family document's grades under (own, other), the other at 3, 3."""

    return [
        document
        for i, (own, other) in enumerate(gaps)
        for document in (
            judged(
                f"p{i}", unit=f"u{i}", subject="phi-writer", grades={"own": own, "other": other}
            ),
            judged(f"q{i}", unit=f"u{i}", grades={"own": 3, "other": 3}),
        )
    ]


def test_a_judge_that_rewards_length_shows_length_bias() -> None:
    documents = [judged(f"d{i}", words=10 * (i + 1), grades={"own": 1 + i // 2}) for i in range(8)]

    result = study(documents, {"own": "solo"})

    (bias,) = result.length
    assert (bias.judge, bias.criterion, bias.controlled) == ("own", "accuracy", False)
    assert bias.rho.value == pytest.approx(1.0, abs=0.05)
    assert bias.evident


def test_length_that_tracks_human_grades_is_not_bias() -> None:
    # Longer documents are better by the human grade, and the judge follows
    # people with noise unrelated to length: alone, rho is high; partialled
    # on the human grade, it vanishes.
    noise = [0, 1, 0, -1, 0, 1, 1, 0, -1, 0, 0, 0]
    documents = [
        judged(f"d{i}", words=10 * (i + 1), grades={"own": 2 + i // 3 + e}, human=1 + i // 3)
        for i, e in enumerate(noise)
    ]

    (bias,) = study(documents, {"own": "solo"}).length

    assert bias.controlled
    assert bias.rho.n == len(noise) >= MIN_CONTROLLED
    assert abs(bias.rho.value) < 0.3
    assert not bias.evident


def test_few_human_ratings_leave_length_bias_uncontrolled() -> None:
    documents = [
        judged(f"d{i}", words=i + 1, grades={"own": i % 5 + 1}, human=3 if i < 2 else None)
        for i in range(6)
    ]

    (bias,) = study(documents, {"own": "solo"}).length

    assert (bias.controlled, bias.rho.n) == (False, 6)


@pytest.mark.parametrize(
    ("documents", "reason"),
    [
        ([judged("a", grades={"own": 3}), judged("b", grades={"own": 4})], "2 graded documents"),
        ([judged(f"d{i}", words=i, grades={"own": 3}) for i in range(4)], "constant"),
    ],
)
def test_length_bias_that_cannot_be_measured_is_skipped(
    documents: list[JudgedDocument], reason: str
) -> None:
    result = study(documents, {"own": "solo"})

    assert result.length == ()
    (skipped,) = result.skipped
    assert skipped.startswith("length bias of own on accuracy")
    assert reason in skipped


def test_a_judge_favouring_its_family_beyond_the_reference_shows_self_preference() -> None:
    # The own-family judge grades its family 2 points up; the reference sees no gap.
    result = study(pairs([(5, 3), (5, 3), (5, 3), (4, 3)]))

    (preference,) = result.self_preference
    assert (preference.judge, preference.family, preference.reference) == ("own", "phi", "other")
    assert preference.effect.value == pytest.approx(1.75)
    assert (preference.judge_gap, preference.reference_gap) == (pytest.approx(1.75), 0)
    assert preference.effect.n == 4
    assert preference.evident


def test_a_gap_both_judges_see_is_not_self_preference() -> None:
    # The phi document really is better: both judges grade it 2 points up.
    (preference,) = study(pairs([(5, 5), (5, 5), (5, 5)])).self_preference

    assert preference.judge_gap == pytest.approx(2)
    assert preference.effect.value == pytest.approx(0)
    assert not preference.evident


def test_self_preference_needs_units_documented_by_both_families() -> None:
    documents = [
        *pairs([(5, 3), (5, 3)]),
        judged("lonely", unit="u9", subject="phi-writer", grades={"own": 5, "other": 3}),
    ]

    result = study(documents)

    assert result.self_preference == ()
    assert "self-preference of own against other on accuracy: 2 units" in result.skipped[-1]


def test_self_preference_needs_a_judge_of_another_family() -> None:
    result = study(pairs([(5, 3)] * 3), {"own": "phi", "twin": "phi"})

    assert result.self_preference == ()
    assert "self-preference of own: no judge of another family" in result.skipped
    assert "self-preference of twin: no judge of another family" in result.skipped


def test_judges_without_a_family_among_the_subjects_are_not_tested() -> None:
    result = study(pairs([(5, 3)] * 3), {"other": "gemma"})

    assert result.self_preference == ()
    assert not any("self-preference" in reason for reason in result.skipped)


def judgments(name: str, judge: str, scores: list[int]) -> list[Judgment]:
    return [judgment(name, "accuracy", s, run=run, judge=judge) for run, s in enumerate(scores)]


def test_judged_documents_take_the_median_of_runs_and_count_words() -> None:
    texts = {"a": "one two three", "b": "unjudged"}
    documents = [
        document(name).model_copy(update={"text": text, "subject": "phi-writer"})
        for name, text in texts.items()
    ]

    (result,) = judged_documents(
        documents,
        judgments("a", "own", [2, 5, 4]) + judgments("a", "other", [3]),
        {doc_id("a"): {"accuracy": 4.0}},
    )

    assert (result.document_id, result.subject, result.words) == (doc_id("a"), "phi-writer", 3)
    assert result.grades == {"own": {"accuracy": 4}, "other": {"accuracy": 3}}
    assert result.human == {"accuracy": 4.0}


def test_human_grades_count_only_the_given_raters_and_rubric() -> None:
    ratings = [
        rating("a", rater="r1", accuracy=2),
        rating("a", rater="r2", accuracy=5),
        rating("a", rater="llm-claude", accuracy=1),
        rating("b", rater="r1", accuracy=3, rubric_hash="e" * 64),
    ]

    grades = human_grades(ratings, raters=("r1", "r2"), rubric_hash=RUBRIC)

    assert grades == {doc_id("a"): {"accuracy": 3.5}}


def test_the_report_marks_evident_bias_and_lists_what_was_skipped() -> None:
    lengthy = [judged(f"d{i}", words=10 * (i + 1), grades={"own": 1 + i // 2}) for i in range(8)]
    result = study(lengthy + pairs([(5, 3), (5, 3), (5, 3), (4, 3)]))

    text = bias_text(result)

    assert "own            accuracy       " in text
    assert "uncontrolled: too few human ratings" in text
    assert "own            vs other          accuracy       +1.750" in text
    assert "EVIDENT" in text
    assert "Not tested\n  length bias of other on accuracy" in text
    assert json.loads(bias_json(result))["self_preference"][0]["judge"] == "own"


def test_an_empty_study_says_so() -> None:
    assert bias_text(BiasStudy((), (), ())) == "no judged documents found"
