from collections.abc import Callable
from typing import Any

import pytest
from scipy import stats

from assay.calibrate import Estimate, RankMethod, rank_correlation

SCORES = [0.91, 0.42, 0.77, 0.15, 0.63, 0.58, 0.30, 0.84, 0.49, 0.70]
RATINGS = [5, 2, 4, 1, 3, 4, 2, 5, 2, 3]
REPOS = ["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4", "r5", "r5"]


@pytest.mark.parametrize(
    ("method", "reference"), [("spearman", stats.spearmanr), ("kendall", stats.kendalltau)]
)
def test_the_point_estimate_matches_scipy(
    method: RankMethod, reference: Callable[..., Any]
) -> None:
    estimate = rank_correlation(SCORES, RATINGS, method=method, resamples=200, seed=1)

    assert estimate.value == pytest.approx(reference(SCORES, RATINGS).statistic)
    assert estimate.ci_low <= estimate.value <= estimate.ci_high
    assert (estimate.n, estimate.confidence) == (10, 0.95)


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_a_perfectly_monotone_sample_gives_one_with_a_degenerate_interval(
    method: RankMethod,
) -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [10.0, 20.0, 25.0, 40.0, 41.0, 90.0]

    estimate = rank_correlation(x, y, method=method, resamples=200, seed=3)

    assert estimate.value == pytest.approx(1.0)
    assert estimate.ci_low == pytest.approx(1.0)
    assert estimate.ci_high == pytest.approx(1.0)


def test_the_same_seed_gives_the_same_interval() -> None:
    def run(seed: int) -> Estimate:
        return rank_correlation(
            SCORES, RATINGS, method="spearman", groups=REPOS, resamples=300, seed=seed
        )

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_clustering_changes_the_interval_but_not_the_value() -> None:
    plain = rank_correlation(SCORES, RATINGS, method="spearman", resamples=300, seed=5)
    clustered = rank_correlation(
        SCORES, RATINGS, method="spearman", groups=REPOS, resamples=300, seed=5
    )

    assert clustered.value == plain.value
    assert (clustered.ci_low, clustered.ci_high) != (plain.ci_low, plain.ci_high)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([1, 2, 3], [1, 2], "equally long"),
        ([1, 2], [1, 2], "at least 3"),
        ([1, 1, 1], [1, 2, 3], "constant"),
        ([1, 2, 3], [4, 4, 4], "constant"),
    ],
)
def test_invalid_inputs_raise(x: list[float], y: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rank_correlation(x, y, method="spearman", resamples=10, seed=0)
