import pytest

from assay.calibrate import RatedDocument, Routing, ablate
from assay.calibrate.ablation import composite
from assay.rating import Decision
from assay.scoring import RoutingDecision

from .test_study import join, rating, score

ROUTING = Routing(publish_at_or_above=70, escalate_below=40)
WEIGHTS = {"accuracy": 0.5, "readability": 0.5}
NOISE = [0.9, 0.1, 0.6, 0.2, 0.5, 0.8, 0.3]


def documents(*, with_reject: bool = True) -> list[RatedDocument]:
    """Accuracy tracks the raters exactly; readability is noise."""

    levels = [1, 2, 3, 4, 5, 2, 4]
    scores = [
        score(
            f"d{i}",
            accuracy=(level - 1) / 4,
            readability=noise,
            composite=50.0,
        )
        for i, (level, noise) in enumerate(zip(levels, NOISE, strict=True))
    ]
    ratings = [
        rating(
            f"d{i}",
            overall=level,
            decision=Decision.PUBLISH if level >= 4 else Decision.REVISE,
        )
        for i, level in enumerate(levels)
    ]
    if with_reject:
        scores.append(
            score(
                "bad",
                accuracy=1.0,
                readability=1.0,
                composite=100.0,
                decision=RoutingDecision.REJECT,
            )
        )
        ratings.append(rating("bad", overall=1, decision=Decision.ESCALATE))
    return join(scores, ratings)


def by_removed(docs: list[RatedDocument], weights: dict[str, float] = WEIGHTS):
    study = ablate(docs, weights=weights, judged={"accuracy"}, routing=ROUTING)
    return study, {a.removed: a for a in study.ablations}


def test_each_component_the_judge_and_the_reject_rule_are_removed_in_turn() -> None:
    _, ablations = by_removed(documents())

    assert list(ablations) == ["accuracy", "readability", "judge", "reject_rule"]


def test_removing_the_informative_component_lowers_agreement() -> None:
    study, ablations = by_removed(documents())

    assert study.full.spearman is not None
    assert 0 < study.full.spearman < 1
    assert ablations["accuracy"].delta_spearman is not None
    assert ablations["accuracy"].delta_spearman < 0
    assert ablations["readability"].outcome.spearman == pytest.approx(1.0)
    assert ablations["readability"].delta_spearman == pytest.approx(1 - study.full.spearman)


def test_rejection_counts_as_escalation_and_the_rule_can_be_ablated() -> None:
    study, ablations = by_removed(documents(), weights={"accuracy": 1.0})

    # d0-d6 have composites 0, 25, 50, 75, 100, 25, 75: escalate, escalate,
    # revise, publish, publish, escalate, publish against human revise, revise,
    # revise, publish, publish, revise, publish, so d2, d3, d4 and d6 match.
    # "bad" is rejected, which matches the human escalate.
    assert study.full.decision_agreement == pytest.approx(5 / 8)
    without_rule = ablations["reject_rule"]
    assert without_rule.outcome.decision_agreement == pytest.approx(4 / 8)
    assert without_rule.delta_decision_agreement == pytest.approx(-1 / 8)
    assert without_rule.delta_spearman is not None
    assert without_rule.delta_spearman < 0  # the rejected document now counts, ranked top


def test_a_document_left_without_components_is_not_scored() -> None:
    docs = documents(with_reject=False)

    _, ablations = by_removed(docs, weights={"accuracy": 1.0})

    assert ablations["accuracy"].outcome.documents == 0
    assert ablations["accuracy"].outcome.spearman is None
    assert ablations["accuracy"].delta_spearman is None
    assert composite(docs[0], {"clarity": 1.0}) is None


def test_no_judge_ablation_without_judged_weights() -> None:
    study = ablate(documents(), weights={"readability": 1.0}, judged={"accuracy"}, routing=ROUTING)

    assert [a.removed for a in study.ablations] == ["readability", "reject_rule"]


@pytest.mark.parametrize(
    ("composite_value", "critical", "decision"),
    [
        (70.0, False, RoutingDecision.PUBLISH),
        (69.9, False, RoutingDecision.REVISE),
        (40.0, False, RoutingDecision.REVISE),
        (39.9, False, RoutingDecision.ESCALATE),
        (99.0, True, RoutingDecision.REJECT),
    ],
)
def test_routing_matches_the_scorer_thresholds(
    composite_value: float, critical: bool, decision: RoutingDecision
) -> None:
    assert ROUTING.route(composite_value, critical=critical) is decision
