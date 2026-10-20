import pytest

from assay.calibrate import Routing, choose_routing, decision_quality
from assay.calibrate.decisions import matches, routed
from assay.rating import Decision
from assay.scoring import RoutingDecision

from .test_study import join, rating, score

P, R, E, X = (
    RoutingDecision.PUBLISH,
    RoutingDecision.REVISE,
    RoutingDecision.ESCALATE,
    RoutingDecision.REJECT,
)


@pytest.mark.parametrize(
    ("scorer", "human", "expected"),
    [
        (P, Decision.PUBLISH, True),
        (R, Decision.PUBLISH, False),
        (X, Decision.ESCALATE, True),
        (X, Decision.REVISE, False),
    ],
)
def test_a_rejection_matches_an_escalation(
    scorer: RoutingDecision, human: Decision, expected: bool
) -> None:
    assert matches(scorer, human) is expected


def test_quality_leaves_out_disputed_decisions() -> None:
    quality = decision_quality(
        [
            (P, Decision.PUBLISH),
            (P, Decision.REVISE),
            (R, Decision.REVISE),
            (E, None),
        ]
    )

    assert quality.decided == 3
    assert quality.agreement == pytest.approx(2 / 3)
    assert quality.published == 2
    assert quality.publish_precision == pytest.approx(1 / 2)


def test_quality_without_decisions_or_publishes_is_undefined() -> None:
    assert decision_quality([(E, None)]).agreement is None
    assert decision_quality([(R, Decision.REVISE)]).publish_precision is None


def documents(pairs: list[tuple[float, Decision]]):
    return join(
        [score(f"d{i}", composite=c) for i, (c, _) in enumerate(pairs)],
        [rating(f"d{i}", decision=d) for i, (_, d) in enumerate(pairs)],
    )


def test_thresholds_are_chosen_to_match_the_raters() -> None:
    docs = documents(
        [
            (10.0, Decision.ESCALATE),
            (25.0, Decision.ESCALATE),
            (45.0, Decision.REVISE),
            (60.0, Decision.REVISE),
            (82.0, Decision.PUBLISH),
            (95.0, Decision.PUBLISH),
        ]
    )

    chosen = choose_routing(docs, step=5)

    assert chosen == Routing(publish_at_or_above=80, escalate_below=45)
    assert routed(docs, chosen).agreement == 1.0


def test_ties_publish_less_and_escalate_more() -> None:
    docs = documents([(50.0, Decision.REVISE)])

    assert choose_routing(docs, step=10) == Routing(publish_at_or_above=100, escalate_below=50)


def test_no_thresholds_without_undisputed_decisions() -> None:
    docs = join(
        [score("a")],
        [rating("a", decision=Decision.PUBLISH), rating("a", rater="r2")],
        raters=("r1", "r2"),
    )

    assert choose_routing(docs) is None
