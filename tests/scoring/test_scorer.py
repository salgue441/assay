from collections.abc import Iterable, Mapping

import pytest

from assay.judge import Judgment
from assay.metrics import MetricRow
from assay.scoring import (
    Component,
    CompositeScorer,
    RoutingDecision,
    Score,
    ScorerConfig,
    ScoringReport,
)

from .fixtures import doc_id, document, judgment, metric_row

WEIGHTS = {"referential_integrity": 0.5, "accuracy": 0.3, "clarity": 0.2}


def run(
    metric_rows: Iterable[MetricRow],
    judgments: Iterable[Judgment] = (),
    *,
    names: Iterable[str] = ("a",),
    weights: Mapping[str, float] = WEIGHTS,
) -> ScoringReport:
    config = ScorerConfig(weights, judge="grader", publish_at_or_above=80, escalate_below=50)
    return CompositeScorer(config).score([document(name) for name in names], metric_rows, judgments)


def only(report: ScoringReport) -> Score:
    assert report.unscored == []
    [score] = report.scores
    return score


def test_weights_static_and_judged_components_into_a_composite() -> None:
    score = only(
        run(
            [metric_row("a", "referential_integrity", 0.9), metric_row("a", "readability", 0.1)],
            [
                judgment("a", "accuracy", 4, run=0),
                judgment("a", "accuracy", 5, run=1),
                judgment("a", "accuracy", 4, run=2),
                judgment("a", "clarity", 3),
                judgment("a", "clarity", 1, run=1, judge="other"),
            ],
        )
    )

    assert score.composite == 77.5
    assert score.components == (
        Component(name="accuracy", value=0.75, weight=0.3),
        Component(name="clarity", value=0.5, weight=0.2),
        Component(name="referential_integrity", value=0.9, weight=0.5),
    )
    assert (score.missing, score.critical, score.critical_reasons) == ((), False, ())
    assert score.decision is RoutingDecision.REVISE
    assert (score.document_id, score.unit_id, score.repo, score.subject, score.judge) == (
        doc_id("a"),
        "pkg.a",
        "owner/repo",
        "writer",
        "grader",
    )


def test_renormalises_over_the_components_a_document_has() -> None:
    score = only(
        run([metric_row("a", "referential_integrity", 0.8)], [judgment("a", "clarity", 5)])
    )

    assert score.composite == pytest.approx(100 * 6 / 7)
    assert score.composite == round(score.composite, 9)
    assert [(c.name, c.weight) for c in score.components] == [
        ("clarity", pytest.approx(2 / 7)),
        ("referential_integrity", pytest.approx(5 / 7)),
    ]
    assert score.missing == ("accuracy",)
    assert score.decision is RoutingDecision.PUBLISH


def test_documents_without_a_weighted_component_are_unscored() -> None:
    report = run(
        [metric_row("a", "readability", 0.9), metric_row("b", "referential_integrity", 1.0)],
        [judgment("c", "accuracy", 5, judge="other")],
        names=("a", "b", "c", "d"),
    )

    assert [s.document_id for s in report.scores] == [doc_id("b")]
    assert report.unscored == [doc_id("a"), doc_id("c"), doc_id("d")]


def test_rows_for_other_documents_are_ignored() -> None:
    report = run([metric_row("z", "referential_integrity", 0.0)], names=("a",))

    assert (report.scores, report.unscored) == ([], [doc_id("a")])


def test_zero_weights_neither_count_nor_go_missing() -> None:
    score = only(
        run(
            [metric_row("a", "referential_integrity", 0.6), metric_row("a", "readability", 0.0)],
            weights={"referential_integrity": 1.0, "readability": 0.0, "accuracy": 0.0},
        )
    )

    assert score.composite == 60
    assert [c.name for c in score.components] == ["referential_integrity"]
    assert score.missing == ()


def test_a_critical_failure_rejects_but_keeps_the_composite() -> None:
    score = only(
        run(
            [
                metric_row("a", "referential_integrity", 0.9, critical=True),
                metric_row("a", "symbol_coverage", 0.2, critical=True),
            ],
            [judgment("a", "accuracy", 5), judgment("a", "clarity", 5)],
        )
    )

    assert score.composite == 95
    assert score.critical
    assert score.critical_reasons == ("referential_integrity", "symbol_coverage")
    assert score.decision is RoutingDecision.REJECT


@pytest.mark.parametrize(
    ("value", "decision"),
    [
        (1.0, RoutingDecision.PUBLISH),
        (0.8, RoutingDecision.PUBLISH),
        (0.79, RoutingDecision.REVISE),
        (0.5, RoutingDecision.REVISE),
        (0.49, RoutingDecision.ESCALATE),
        (0.0, RoutingDecision.ESCALATE),
    ],
)
def test_routes_at_each_threshold(value: float, decision: RoutingDecision) -> None:
    score = only(run([metric_row("a", "referential_integrity", value)]))

    assert score.composite == 100 * value
    assert score.decision is decision


def test_rounding_noise_does_not_move_a_composite_below_a_threshold() -> None:
    # Unrounded, these give 49.99999999999999 for an exact 50.
    score = only(
        run(
            [metric_row("a", "readability", 0.6)],
            [judgment("a", "accuracy", 1), judgment("a", "clarity", 2)],
            weights={"accuracy": 0.05, "clarity": 0.2, "readability": 0.75},
        )
    )

    assert score.composite == 50
    assert score.decision is RoutingDecision.REVISE


def test_scores_carry_the_scorer_version() -> None:
    config = ScorerConfig(WEIGHTS, judge="grader", publish_at_or_above=80, escalate_below=50)
    report = CompositeScorer(config).score(
        [document("a")], [metric_row("a", "referential_integrity", 1.0)], []
    )

    assert only(report).scorer_version == config.version


def test_duplicate_metric_rows_are_rejected() -> None:
    rows = [
        metric_row("a", "referential_integrity", 0.9),
        metric_row("a", "referential_integrity", 0.8, version="2"),
    ]

    with pytest.raises(ValueError, match="one metric version"):
        run(rows)


def test_a_name_that_is_both_a_metric_and_a_criterion_is_rejected() -> None:
    with pytest.raises(ValueError, match="accuracy is both a metric and a judged criterion"):
        run([metric_row("a", "accuracy", 0.9)], [judgment("a", "accuracy", 5)])
