"""
Rank correlation between the composite score and human ratings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from assay.calibrate.bootstrap import Estimate, bootstrap_interval

RankMethod = Literal["spearman", "kendall"]

MIN_PAIRS = 3


def rank_correlation(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: RankMethod,
    groups: Sequence[str] | None = None,
    resamples: int = 10_000,
    seed: int,
    confidence: float = 0.95,
) -> Estimate:
    """
    Spearman's rho or Kendall's tau-b between `x` and `y`, with a percentile
    bootstrap interval. Given `groups` (one label per pair, e.g. the
    repository), the bootstrap resamples whole groups.
    """

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape or xs.ndim != 1:
        msg = f"x and y must be 1-D and equally long, got {xs.shape} and {ys.shape}"
        raise ValueError(msg)
    if len(xs) < MIN_PAIRS:
        msg = f"rank correlation needs at least {MIN_PAIRS} pairs, got {len(xs)}"
        raise ValueError(msg)
    value = _correlation(xs, ys, method)
    if np.isnan(value):
        msg = "rank correlation is undefined for a constant input"
        raise ValueError(msg)

    def statistic(rows: NDArray[np.intp]) -> float:
        return _correlation(xs[rows], ys[rows], method)

    low, high = bootstrap_interval(
        statistic, len(xs), groups=groups, resamples=resamples, seed=seed, confidence=confidence
    )
    return Estimate(value=value, ci_low=low, ci_high=high, n=len(xs), confidence=confidence)


def _correlation(xs: NDArray[np.float64], ys: NDArray[np.float64], method: RankMethod) -> float:
    """The coefficient, or NaN when either side is constant."""

    if np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(xs, ys).statistic)
    if method == "kendall":
        return float(stats.kendalltau(xs, ys).statistic)
    msg = f"unknown rank correlation method: {method!r}"
    raise ValueError(msg)
