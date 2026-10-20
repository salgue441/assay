import pytest

from assay.calibrate.decisions import Routing
from assay.calibrate.variance import VarianceStudy, judge_variance
from assay.judge import Judgment
from assay.scoring import RoutingDecision, Score

from ..scoring.fixtures import judgment
from .test_study import score

CRITERIA = ("accuracy", "clarity")
WEIGHTS = {"readability": 0.5, "accuracy": 0.25, "clarity": 0.25}
ROUTING = Routing(publish_at_or_above=70, escalate_below=40)


def grades(name: str, runs: list[tuple[int, int]], *, temperature: float = 0.0) -> list[Judgment]:
    """`runs` holds (accuracy, clarity) per run."""

    return [
        judgment(name, criterion, grade, run=run).model_copy(
            update={"temperature": temperature, "judgment_id": f"{run}{criterion}{temperature}"}
        )
        for run, pair in enumerate(runs)
        for criterion, grade in zip(CRITERIA, pair, strict=True)
    ]


def study(scores: list[Score], judgments: list[Judgment]) -> list[VarianceStudy]:
    return judge_variance(scores, judgments, weights=WEIGHTS, criteria=CRITERIA, routing=ROUTING)


def test_identical_runs_have_no_spread_and_keep_their_decision() -> None:
    (result,) = study([score("a", readability=1.0)], grades("a", [(5, 5), (5, 5), (5, 5)]))

    assert result.runs == 3
    assert [(s.criterion, s.mean_sd, s.changed) for s in result.criteria] == [
        ("accuracy", 0.0, 0.0),
        ("clarity", 0.0, 0.0),
    ]
    assert (result.band_stability, result.documents, result.unstable) == (1.0, 1, ())


def test_a_grade_change_that_crosses_a_threshold_is_unstable() -> None:
    # readability 0.5 -> 25 points; accuracy and clarity add 0-25 each.
    # Run 0: grades 5, 5 -> 75 (publish); run 1: grades 3, 3 -> 50 (revise).
    scores = [score("a", readability=0.5), score("b", readability=0.5)]
    judgments = grades("a", [(5, 5), (3, 3)]) + grades("b", [(3, 3), (3, 4)])

    (result,) = study(scores, judgments)

    assert result.band_stability == pytest.approx(0.5)
    (unstable,) = result.unstable
    assert unstable.unit_id == "a"
    assert unstable.decisions == (RoutingDecision.PUBLISH, RoutingDecision.REVISE)
    accuracy = result.criteria[0]
    assert accuracy.mean_sd == pytest.approx(0.5)  # sd 1.0 for a, 0.0 for b
    assert accuracy.changed == pytest.approx(0.5)


def test_rejected_documents_stay_rejected() -> None:
    scores = [score("a", decision=RoutingDecision.REJECT)]

    (result,) = study(scores, grades("a", [(5, 5), (1, 1)]))

    assert result.band_stability == 1.0


def test_each_temperature_is_its_own_study() -> None:
    judgments = grades("a", [(4, 4), (4, 4)]) + grades("a", [(2, 5), (5, 2)], temperature=0.7)

    cold, warm = study([score("a")], judgments)

    assert (cold.temperature, warm.temperature) == (0.0, 0.7)
    assert cold.criteria[0].max_sd == 0.0
    assert warm.criteria[0].max_sd == pytest.approx(1.5)


def test_single_runs_and_incomplete_runs_give_no_stability() -> None:
    incomplete = [*grades("a", [(4, 4)]), judgment("a", "accuracy", 3, run=1)]

    (result,) = study([score("a")], incomplete)

    assert result.band_stability is None
    assert result.documents == 0
    assert result.criteria[0].documents == 1  # accuracy was graded twice


def test_judgments_of_unscored_documents_are_ignored() -> None:
    assert study([score("a")], grades("z", [(1, 1), (5, 5)])) == []
