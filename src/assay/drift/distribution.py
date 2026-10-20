"""Distribution-shift statistics: compare a current score sample with a baseline sample.

A shift is only reported when it is both statistically significant (two-sample
Kolmogorov-Smirnov test) and practically large (Population Stability Index above a floor).
With large samples the KS test flags shifts too small to matter, and with small samples the
PSI is noisy; requiring both keeps the drift monitor quiet unless a change is real and big.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy import stats

from assay.drift._samples import FloatArray, as_sample


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-sample hypothesis test."""

    # Stops pytest from trying to collect this class when a test module imports it.
    __test__: ClassVar[bool] = False

    statistic: float
    p_value: float


@dataclass(frozen=True)
class ShiftResult:
    """Combined verdict of the KS test and the PSI for one baseline/current comparison."""

    ks: TestResult
    psi: float
    breached: bool


def ks_test(baseline: npt.ArrayLike, current: npt.ArrayLike) -> TestResult:
    """Run a two-sided two-sample Kolmogorov-Smirnov test of ``current`` against ``baseline``.

    Raises ``ValueError`` if either sample is empty or contains non-finite values.
    """
    reference = as_sample(baseline, name="baseline")
    observed = as_sample(current, name="current")
    result = stats.ks_2samp(reference, observed, alternative="two-sided")
    return TestResult(statistic=float(result.statistic), p_value=float(result.pvalue))


def population_stability_index(
    baseline: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Return the Population Stability Index of ``current`` relative to ``baseline``.

    Bin edges are the interior ``1/bins, 2/bins, ...`` quantiles of the baseline. When the
    baseline has ties (for example discrete rubric scores), several quantiles coincide; the
    duplicates are dropped, so the result may use fewer than ``bins`` bins. Bins are closed
    on the right, ``(edge[i-1], edge[i]]``, matching the quantile definition
    ``P(X <= q) >= p``, and the outer bins are open-ended so every current value lands in a
    bin. Bin proportions are floored at ``epsilon`` so that empty bins keep the logarithm
    finite. PSI is ``sum((c - b) * ln(c / b))`` over the bins, where ``b`` and ``c`` are the
    baseline and current proportions; it is zero for identical distributions.

    Raises ``ValueError`` for empty or non-finite samples, ``bins < 2`` or ``epsilon <= 0``.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    reference = as_sample(baseline, name="baseline")
    observed = as_sample(current, name="current")

    edges = _quantile_edges(reference, bins)
    expected = np.maximum(_bin_proportions(reference, edges), epsilon)
    actual = np.maximum(_bin_proportions(observed, edges), epsilon)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def distribution_shift(
    baseline: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    ks_alpha: float,
    psi_floor: float,
) -> ShiftResult:
    """Compare ``current`` with ``baseline`` and decide whether the shift breaches the limits.

    ``breached`` is true only when the KS p-value is below ``ks_alpha`` (the shift is
    significant) and the PSI is at least ``psi_floor`` (the shift is large enough to matter).

    Raises ``ValueError`` if ``ks_alpha`` is not in (0, 1), ``psi_floor`` is negative, or a
    sample is invalid.
    """
    if not 0 < ks_alpha < 1:
        raise ValueError(f"ks_alpha must be in (0, 1), got {ks_alpha}")
    if psi_floor < 0:
        raise ValueError(f"psi_floor must be non-negative, got {psi_floor}")
    ks = ks_test(baseline, current)
    psi = population_stability_index(baseline, current)
    return ShiftResult(ks=ks, psi=psi, breached=ks.p_value < ks_alpha and psi >= psi_floor)


def _quantile_edges(reference: FloatArray, bins: int) -> FloatArray:
    """Return the sorted, de-duplicated interior quantile edges of ``reference``."""
    probabilities = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.unique(np.quantile(reference, probabilities))


def _bin_proportions(sample: FloatArray, edges: FloatArray) -> FloatArray:
    """Return the share of ``sample`` in each right-closed bin delimited by ``edges``."""
    indices = np.searchsorted(edges, sample, side="left")
    counts = np.bincount(indices, minlength=edges.size + 1)
    return counts.astype(np.float64) / sample.size
