"""Failure-rate test: decide whether an observed failure rate exceeds its baseline rate."""

from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class RateTest:
    """Observed failure rate and the one-sided exact test against the baseline rate."""

    rate: float
    p_value: float
    exceeds: bool


def binomial_rate_test(
    failures: int,
    trials: int,
    *,
    baseline_rate: float,
    alpha: float,
) -> RateTest:
    """Test whether ``failures`` out of ``trials`` exceeds ``baseline_rate``.

    Runs a one-sided exact binomial test (alternative: the true rate is greater than the
    baseline). ``exceeds`` is true when the p-value is below ``alpha``.

    Raises ``ValueError`` if ``trials < 1``, ``failures`` is not in ``[0, trials]``,
    ``baseline_rate`` is not in [0, 1] or ``alpha`` is not in (0, 1).
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if not 0 <= failures <= trials:
        raise ValueError(f"failures must be in [0, {trials}], got {failures}")
    if not 0 <= baseline_rate <= 1:
        raise ValueError(f"baseline_rate must be in [0, 1], got {baseline_rate}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    result = stats.binomtest(failures, trials, p=baseline_rate, alternative="greater")
    p_value = float(result.pvalue)
    return RateTest(rate=failures / trials, p_value=p_value, exceeds=p_value < alpha)
