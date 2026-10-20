import hashlib
from datetime import UTC, datetime

import pytest

from assay.calibrate import (
    CalibrationReport,
    RatedDocument,
    Routing,
    calibrate,
    rated_documents,
)
from assay.rating import CriterionScore, Decision, Rating, rating_id
from assay.scoring import Component, RoutingDecision, Score

COMMIT = "c" * 40
SCORER = "a" * 16
RUBRIC = "b" * 64
PRIOR = {"readability": 0.5, "accuracy": 0.5}
ROUTING = Routing(publish_at_or_above=70, escalate_below=40)


def doc_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def score(
    name: str,
    *,
    repo: str = "o/one",
    composite: float = 50.0,
    accuracy: float | None = 0.5,
    readability: float = 0.5,
    decision: RoutingDecision = RoutingDecision.REVISE,
    scorer_version: str = SCORER,
) -> Score:
    components = [Component(name="readability", value=readability, weight=0.5)]
    if accuracy is not None:
        components.append(Component(name="accuracy", value=accuracy, weight=0.5))
    rejected = decision is RoutingDecision.REJECT
    return Score(
        document_id=doc_id(name),
        unit_id=name,
        repo=repo,
        commit=COMMIT,
        subject="writer",
        scorer_version=scorer_version,
        judge="grader",
        composite=composite,
        components=tuple(components),
        missing=(),
        critical=rejected,
        critical_reasons=("referential_integrity",) if rejected else (),
        decision=decision,
    )


def rating(
    name: str,
    *,
    rater: str = "r1",
    overall: int = 3,
    accuracy: int = 3,
    decision: Decision = Decision.REVISE,
    rubric_hash: str = RUBRIC,
) -> Rating:
    return Rating(
        rating_id=rating_id(document_id=doc_id(name), rater_id=rater, rubric_hash=rubric_hash),
        document_id=doc_id(name),
        unit_id=name,
        repo="o/one",
        commit=COMMIT,
        rater_id=rater,
        rubric_version="v1",
        rubric_hash=rubric_hash,
        scores=(CriterionScore(criterion="accuracy", score=accuracy),),
        overall=overall,
        decision=decision,
        notes="",
        seconds_spent=1.0,
        rated_at=datetime(2026, 10, 20, tzinfo=UTC),
    )


def join(
    scores: list[Score], ratings: list[Rating], raters: tuple[str, ...] = ("r1",)
) -> list[RatedDocument]:
    return rated_documents(
        scores, ratings, raters=raters, scorer_version=SCORER, rubric_hash=RUBRIC
    )


def test_only_selected_raters_the_current_rubric_and_scorer_are_joined() -> None:
    scores = [score("a"), score("b"), score("c", scorer_version="f" * 16)]
    ratings = [
        rating("a"),
        rating("b", rater="llm-claude"),
        rating("c"),
        rating("a", rater="r2", rubric_hash="e" * 64),
    ]

    documents = join(scores, ratings)

    assert [d.unit_id for d in documents] == ["a"]
    assert list(documents[0].ratings) == ["r1"]


def test_raters_are_combined_into_a_consensus() -> None:
    ratings = [
        rating("a", overall=2, accuracy=2, decision=Decision.REVISE),
        rating("a", rater="r2", overall=5, accuracy=3, decision=Decision.PUBLISH),
    ]

    (document,) = join([score("a")], ratings, raters=("r1", "r2"))

    assert document.overall == 3.5
    assert document.criteria == {"accuracy": 3}  # median 2.5, rounded half up
    assert document.human_decision is None
    assert document.ratings["r2"] == {"overall": 5, "accuracy": 3}


@pytest.mark.parametrize(("value", "grade"), [(0.0, 1), (0.5, 3), (0.625, 4), (1.0, 5)])
def test_the_judge_grade_is_read_back_onto_the_rubric_scale(value: float, grade: int) -> None:
    (document,) = join([score("a", accuracy=value)], [rating("a")])

    assert document.judge_grade("accuracy") == grade
    assert document.judge_grade("clarity") is None


def corpus(raters: tuple[str, ...] = ("r1",)) -> list[RatedDocument]:
    """Four repositories whose composites order documents as the raters do,
    plus one rejected document the raters dislike."""

    scores, ratings = [], []
    for r, repo in enumerate(["o/a", "o/b", "o/c", "o/d"]):
        for i in range(4):
            name = f"{repo}-{i}"
            level = (i + r) % 5 + 1
            scores.append(
                score(
                    name,
                    repo=repo,
                    composite=20.0 * level - 10,
                    accuracy=(level - 1) / 4,
                    readability=(level - 1) / 4,
                    decision=RoutingDecision.PUBLISH if level >= 4 else RoutingDecision.REVISE,
                )
            )
            for rater in raters:
                ratings.append(
                    rating(
                        name,
                        rater=rater,
                        overall=level,
                        accuracy=level,
                        decision=Decision.PUBLISH if level >= 4 else Decision.REVISE,
                    )
                )
    scores.append(
        score("bad", repo="o/a", composite=95.0, decision=RoutingDecision.REJECT, accuracy=0.0)
    )
    ratings.extend(
        rating("bad", rater=rater, overall=1, accuracy=1, decision=Decision.ESCALATE)
        for rater in raters
    )
    return join(scores, ratings, raters=raters)


def texts(documents: list[RatedDocument]) -> dict[str, str]:
    """A text per document whose length grows with its human rating."""

    return {
        d.document_id: "## Summary\n" + " ".join(["word"] * int(10 * d.overall)) for d in documents
    }


def run(documents: list[RatedDocument], *, include_test: bool = False) -> CalibrationReport:
    return calibrate(
        documents,
        prior=PRIOR,
        routing=ROUTING,
        development_fraction=0.5,
        shrink_to_prior=True,
        resamples=50,
        seed=7,
        include_test=include_test,
        texts=texts(documents),
    )


def test_a_score_that_orders_documents_like_the_raters_correlates_perfectly() -> None:
    report = run(corpus(), include_test=True)

    assert set(report.dev_repos).isdisjoint(report.test_repos)
    assert report.dev_documents + report.test_documents == report.documents == 17
    for side in ("dev", "test"):
        assert report.correlation[side]["spearman"].value == pytest.approx(1.0)
        assert report.correlation[side]["kendall"].value == pytest.approx(1.0)
        assert report.judge_kappa[side]["accuracy"].value == pytest.approx(1.0)


def test_the_test_split_is_reported_only_on_request() -> None:
    report = run(corpus())

    assert set(report.correlation) == set(report.judge_kappa) == {"dev"}
    assert set(report.decisions) == set(report.ablations) == {"dev"}


def test_rejected_documents_are_left_out_of_correlations_but_counted_as_decisions() -> None:
    report = run(corpus(), include_test=True)
    side = "dev" if "o/a" in report.dev_repos else "test"

    assert report.correlation[side]["spearman"].n == 8
    assert report.decisions[side]["reject"] == {"escalate": 1}


def test_weights_are_fitted_on_development_repositories() -> None:
    report = run(corpus())

    assert report.fitted is not None
    assert sum(report.fitted.weights.values()) == pytest.approx(1.0)
    assert report.fitted.documents == report.dev_documents
    assert report.fitted_correlation["dev"]["spearman"].value == pytest.approx(1.0)


def test_one_development_repository_keeps_the_prior_weights_and_says_why() -> None:
    documents = [d for d in corpus() if d.repo in {"o/a", "o/b"}]

    report = run(documents)

    assert report.fitted is None
    assert report.fitted_correlation == {}
    assert any("at least two development repositories" in note for note in report.notes)


def test_one_repository_bootstraps_documents_instead_of_collapsing_the_interval() -> None:
    documents = join(
        [score(f"d{i}", repo="o/x" if i < 8 else "o/y", composite=6.0 * i) for i in range(16)],
        [rating(f"d{i}", overall=[1, 3, 2, 4, 3, 5, 4, 2][i % 8]) for i in range(16)],
    )

    report = run(documents)
    rho = report.correlation["dev"]["spearman"]

    assert rho.ci_low < rho.value < rho.ci_high
    assert any("dev intervals: one repository" in note for note in report.notes)


def test_rater_agreement_needs_two_raters() -> None:
    assert run(corpus()).rater_alpha == {}

    alpha = run(corpus(raters=("r1", "r2"))).rater_alpha

    assert alpha == {"overall": pytest.approx(1.0), "accuracy": pytest.approx(1.0)}


def test_an_undefined_statistic_is_explained_instead_of_raised() -> None:
    documents = join(
        [score(f"d{i}", repo=f"o/{i % 2}", composite=10.0 * i) for i in range(6)],
        [rating(f"d{i}", overall=3) for i in range(6)],
    )

    report = run(documents)

    assert "spearman" not in report.correlation["dev"]
    assert any("constant input" in note for note in report.notes)


def test_the_largest_rank_gaps_come_first() -> None:
    """Each repository has one document the score ranks last and raters best,
    so the result does not depend on which repository is held out."""

    scores, ratings = [], []
    for repo in ("o/x", "o/y"):
        for i, overall in enumerate([1, 2, 3, 4]):
            scores.append(score(f"{repo}-{i}", repo=repo, composite=20.0 + 10 * i))
            ratings.append(rating(f"{repo}-{i}", overall=overall))
        scores.append(score(f"{repo}-odd", repo=repo, composite=0.0))
        ratings.append(rating(f"{repo}-odd", overall=5))

    report = run(join(scores, ratings))
    (dev_repo,) = report.dev_repos

    gaps = [abs(g.rank_gap) for g in report.disagreements]
    assert gaps == sorted(gaps, reverse=True)
    assert report.disagreements[0].unit_id == f"{dev_repo}-odd"
    assert report.disagreements[0].rank_gap == pytest.approx(-1.0)
    assert {g.repo for g in report.disagreements} == {dev_repo}


def test_thresholds_are_chosen_on_development_and_judged_on_each_split() -> None:
    report = run(corpus(), include_test=True)

    assert report.chosen_routing is not None
    for side in ("dev", "test"):
        assert set(report.decision_quality[side]) == {"configured", "chosen"}
        assert sum(sum(h.values()) for h in report.chosen_decisions[side].values()) == sum(
            sum(h.values()) for h in report.decisions[side].values()
        )
    assert report.decision_quality["dev"]["chosen"].agreement == 1.0


def test_stratifying_needs_every_document_to_have_the_variables() -> None:
    documents = corpus()

    with pytest.raises(ValueError, match="documents without language"):
        calibrate(
            documents,
            prior=PRIOR,
            routing=ROUTING,
            development_fraction=0.5,
            shrink_to_prior=False,
            resamples=20,
            seed=7,
            include_test=False,
            texts=texts(documents),
            stratify_by=("language",),
        )


def test_the_split_balances_the_stratum_variables() -> None:
    scores, ratings, strata = [], [], {}
    for r, (repo, lang) in enumerate(
        [("o/p1", "python"), ("o/p2", "python"), ("o/g1", "go"), ("o/g2", "go")]
    ):
        for i in range(4):
            name = f"{repo}-{i}"
            scores.append(score(name, repo=repo, composite=10.0 * (i + r)))
            ratings.append(rating(name, overall=(i + r) % 5 + 1))
            strata[doc_id(name)] = {"language": lang}
    documents = rated_documents(
        scores, ratings, raters=("r1",), scorer_version=SCORER, rubric_hash=RUBRIC, strata=strata
    )

    report = calibrate(
        documents,
        prior=PRIOR,
        routing=ROUTING,
        development_fraction=0.5,
        shrink_to_prior=False,
        resamples=20,
        seed=7,
        include_test=False,
        texts=texts(documents),
        stratify_by=("language",),
    )

    test_languages = sorted("go" if "/g" in r else "python" for r in report.test_repos)
    assert test_languages == ["go", "python"]
