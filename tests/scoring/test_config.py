import pytest

from assay.scoring import RoutingDecision, ScorerConfig

WEIGHTS = {"referential_integrity": 0.5, "accuracy": 0.3, "clarity": 0.2}


def config(**changes: object) -> ScorerConfig:
    values: dict[str, object] = {
        "weights": WEIGHTS,
        "judge": "grader",
        "publish_at_or_above": 80,
        "escalate_below": 50,
    } | changes
    return ScorerConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({}, "at least one"),
        ({"a": 1.2, "b": -0.2}, "must not be negative: b"),
        ({"a": 0.5, "b": 0.4}, "sum to 1, got 0.900000"),
        ({"a": 0.5, "b": 0.5 + 1e-6}, "sum to 1"),
    ],
)
def test_rejects_invalid_weights(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config(weights=weights)


def test_weights_within_rounding_of_one_are_accepted() -> None:
    assert config(weights={"a": 0.1, "b": 0.2, "c": 0.7}).weights["c"] == 0.7


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"publish_at_or_above": 101}, "publish_at_or_above must be within 0-100"),
        ({"escalate_below": -1}, "escalate_below must be within 0-100"),
        ({"publish_at_or_above": 40, "escalate_below": 60}, "must not exceed"),
        ({"judge": ""}, "judge"),
    ],
)
def test_rejects_invalid_thresholds_and_judge(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config(**changes)


def test_equal_thresholds_leave_no_revise_band() -> None:
    scorer = config(publish_at_or_above=60, escalate_below=60)

    assert scorer.route(60, critical=False) is RoutingDecision.PUBLISH
    assert scorer.route(59.999, critical=False) is RoutingDecision.ESCALATE


def test_keeps_a_copy_of_the_weights() -> None:
    weights = dict(WEIGHTS)
    scorer = config(weights=weights)
    version = scorer.version
    weights["accuracy"] = 0.0

    assert scorer.weights["accuracy"] == 0.3
    assert config(weights=dict(WEIGHTS)).version == version


@pytest.mark.parametrize(
    ("composite", "decision"),
    [
        (100, RoutingDecision.PUBLISH),
        (80, RoutingDecision.PUBLISH),
        (79.999, RoutingDecision.REVISE),
        (50, RoutingDecision.REVISE),
        (49.999, RoutingDecision.ESCALATE),
        (0, RoutingDecision.ESCALATE),
    ],
)
def test_routes_on_the_thresholds(composite: float, decision: RoutingDecision) -> None:
    assert config().route(composite, critical=False) is decision


def test_a_critical_failure_is_rejected_whatever_the_composite() -> None:
    assert config().route(100, critical=True) is RoutingDecision.REJECT


def test_version_is_a_stable_short_hash() -> None:
    version = config().version

    assert len(version) == 16
    assert int(version, 16) >= 0
    assert version == config().version
    assert version == config(weights=dict(reversed(WEIGHTS.items()))).version
    assert version == config(publish_at_or_above=80.0, escalate_below=50.0).version


@pytest.mark.parametrize(
    "changes",
    [
        {"weights": {"referential_integrity": 0.4, "accuracy": 0.4, "clarity": 0.2}},
        {"weights": {"referential_integrity": 0.5, "accuracy": 0.3, "readability": 0.2}},
        {"judge": "other"},
        {"publish_at_or_above": 81},
        {"escalate_below": 49},
    ],
)
def test_version_changes_with_any_setting(changes: dict[str, object]) -> None:
    assert config(**changes).version != config().version
