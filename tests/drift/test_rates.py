import pytest
from scipy import stats

from assay.drift import binomial_rate_test


def test_all_failures_at_even_odds_has_known_p_value() -> None:
    # P(X >= 3 | n = 3, p = 0.5) = 0.5 ** 3.
    result = binomial_rate_test(3, 3, baseline_rate=0.5, alpha=0.05)

    assert result.rate == 1.0
    assert result.p_value == pytest.approx(0.125)
    assert not result.exceeds


def test_p_value_matches_scipy_one_sided_test() -> None:
    result = binomial_rate_test(9, 20, baseline_rate=0.2, alpha=0.05)

    expected = stats.binomtest(9, 20, p=0.2, alternative="greater").pvalue
    assert result.rate == pytest.approx(0.45)
    assert result.p_value == pytest.approx(expected)
    assert result.p_value == pytest.approx(stats.binom.sf(8, 20, 0.2))
    assert result.exceeds


def test_rate_at_baseline_does_not_exceed() -> None:
    result = binomial_rate_test(20, 100, baseline_rate=0.2, alpha=0.05)

    assert result.p_value > 0.05
    assert not result.exceeds


def test_zero_failures_have_p_value_one() -> None:
    result = binomial_rate_test(0, 10, baseline_rate=0.1, alpha=0.05)

    assert result.rate == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert not result.exceeds


@pytest.mark.parametrize(
    ("failures", "trials", "baseline_rate", "alpha"),
    [
        (0, 0, 0.1, 0.05),
        (-1, 10, 0.1, 0.05),
        (11, 10, 0.1, 0.05),
        (1, 10, -0.1, 0.05),
        (1, 10, 1.1, 0.05),
        (1, 10, 0.1, 0.0),
        (1, 10, 0.1, 1.0),
    ],
)
def test_binomial_rate_test_rejects_invalid_inputs(
    failures: int, trials: int, baseline_rate: float, alpha: float
) -> None:
    with pytest.raises(ValueError):
        binomial_rate_test(failures, trials, baseline_rate=baseline_rate, alpha=alpha)
