import math

import numpy as np
import pytest
from scipy import stats

from assay.drift import distribution_shift, ks_test, population_stability_index


def _normal(mean: float, size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 1.0, size)


def test_ks_test_matches_scipy() -> None:
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
    current = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    result = ks_test(baseline, current)

    expected = stats.ks_2samp(baseline, current)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)


@pytest.mark.parametrize(("baseline", "current"), [([], [1.0]), ([1.0], [])])
def test_ks_test_rejects_empty_samples(baseline: list[float], current: list[float]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ks_test(baseline, current)


def test_ks_test_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ks_test([1.0, math.nan], [1.0])


def test_psi_is_zero_for_identical_samples() -> None:
    sample = _normal(0.0, 500, seed=1)

    assert population_stability_index(sample, sample) == pytest.approx(0.0)


def test_psi_matches_hand_computed_value() -> None:
    # Two bins split at the baseline median 5.5: baseline 50/50, current 30/70.
    baseline = [float(v) for v in range(1, 11)]
    current = [1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

    psi = population_stability_index(baseline, current, bins=2)

    expected = (0.3 - 0.5) * math.log(0.3 / 0.5) + (0.7 - 0.5) * math.log(0.7 / 0.5)
    assert psi == pytest.approx(expected)
    assert psi == pytest.approx(0.2 * math.log(7 / 3))


def test_psi_floors_empty_bins_at_epsilon() -> None:
    # Edge at 1.5: the current sample leaves the lower bin empty.
    psi = population_stability_index([1.0, 2.0], [2.0, 2.0], bins=2, epsilon=0.01)

    expected = (0.01 - 0.5) * math.log(0.01 / 0.5) + (1.0 - 0.5) * math.log(1.0 / 0.5)
    assert psi == pytest.approx(expected)


def test_psi_open_outer_bins_catch_values_outside_the_baseline_range() -> None:
    baseline = [float(v) for v in range(1, 11)]
    far_below = [-100.0] * 10

    psi = population_stability_index(baseline, far_below, bins=2)

    expected = (1.0 - 0.5) * math.log(1.0 / 0.5) + (1e-4 - 0.5) * math.log(1e-4 / 0.5)
    assert psi == pytest.approx(expected)


def test_psi_tied_baseline_merges_duplicate_edges() -> None:
    # Every interior quartile is 3, so the edges collapse to one: bins (-inf, 3] and (3, inf).
    baseline = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0]
    current = [3.0] * 5 + [5.0] * 5

    psi = population_stability_index(baseline, current, bins=4)

    expected = (0.5 - 0.9) * math.log(0.5 / 0.9) + (0.5 - 0.1) * math.log(0.5 / 0.1)
    assert psi == pytest.approx(expected)


def test_psi_constant_baseline_still_detects_upward_shift() -> None:
    psi = population_stability_index([3.0] * 20, [5.0] * 20)

    assert psi == pytest.approx(2 * (1 - 1e-4) * math.log(1 / 1e-4))


@pytest.mark.parametrize(("bins", "epsilon"), [(1, 1e-4), (10, 0.0), (10, -1.0)])
def test_psi_rejects_invalid_parameters(bins: int, epsilon: float) -> None:
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0], [1.0, 2.0], bins=bins, epsilon=epsilon)


def test_identical_samples_do_not_breach() -> None:
    sample = _normal(0.0, 500, seed=2)

    result = distribution_shift(sample, sample, ks_alpha=0.05, psi_floor=0.1)

    assert result.ks.p_value == pytest.approx(1.0)
    assert result.psi == pytest.approx(0.0)
    assert not result.breached


def test_clearly_shifted_sample_breaches() -> None:
    baseline = _normal(0.0, 500, seed=3)
    current = _normal(1.0, 500, seed=4)

    result = distribution_shift(baseline, current, ks_alpha=0.05, psi_floor=0.1)

    assert result.ks.p_value < 0.05
    assert result.psi >= 0.1
    assert result.breached


def test_large_sample_tiny_shift_is_significant_but_not_breached() -> None:
    # With 20k scores per side a 0.05 sd shift is detectable but far below the PSI floor,
    # which is why a breach requires both significance and effect size.
    baseline = _normal(0.0, 20_000, seed=5)
    current = _normal(0.05, 20_000, seed=6)

    result = distribution_shift(baseline, current, ks_alpha=0.05, psi_floor=0.1)

    assert result.ks.p_value < 0.05
    assert result.psi < 0.1
    assert not result.breached


@pytest.mark.parametrize(("ks_alpha", "psi_floor"), [(0.0, 0.1), (1.0, 0.1), (0.05, -0.1)])
def test_distribution_shift_rejects_invalid_thresholds(ks_alpha: float, psi_floor: float) -> None:
    with pytest.raises(ValueError):
        distribution_shift([1.0, 2.0], [1.0, 2.0], ks_alpha=ks_alpha, psi_floor=psi_floor)


def test_ks_test_rejects_multidimensional_samples() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ks_test([[1.0, 2.0]], [1.0])
