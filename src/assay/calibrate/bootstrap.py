"""
Percentile bootstrap confidence intervals, optionally clustered by group.

Ratings from one repository are not independent, so the calibration study
resamples whole repositories rather than single documents whenever the
caller says which repository each pair came from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_VALID_SHARE = 0.5
"""Smallest share of bootstrap replicates that must give a defined statistic."""

Statistic = Callable[[NDArray[np.intp]], float]
"""A statistic computed on the rows a replicate picked; NaN when undefined."""


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a percentile bootstrap confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    n: int
    """Pairs the point estimate was computed on."""
    confidence: float
    """Nominal coverage of the interval, e.g. 0.95."""


def resample_indices(
    n: int, groups: Sequence[str] | None, rng: np.random.Generator
) -> NDArray[np.intp]:
    """
    Row indices for one bootstrap replicate. Without groups, `n` rows are
    drawn with replacement. With groups, as many groups as there are are
    drawn with replacement and every row of each drawn group is kept, so a
    group is never split.
    """

    if groups is None:
        return rng.integers(0, n, size=n).astype(np.intp)
    members = _members(groups)
    picked = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in picked])


def bootstrap_interval(
    statistic: Statistic,
    n: int,
    *,
    groups: Sequence[str] | None,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """
    The percentile interval of `statistic` over `resamples` replicates of
    `n` rows. Replicates where the statistic is undefined (NaN) are skipped;
    fewer than `MIN_VALID_SHARE` of them defined raises ValueError.
    """

    if resamples < 1:
        msg = f"resamples must be at least 1, got {resamples}"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1), got {confidence}"
        raise ValueError(msg)
    if groups is not None and len(groups) != n:
        msg = f"groups has {len(groups)} entries for {n} pairs"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    values = np.array([statistic(resample_indices(n, groups, rng)) for _ in range(resamples)])
    defined = values[~np.isnan(values)]
    if len(defined) < max(1, math.ceil(MIN_VALID_SHARE * resamples)):
        msg = f"only {len(defined)} of {resamples} bootstrap replicates were defined"
        raise ValueError(msg)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(defined, [tail, 1.0 - tail])
    return float(low), float(high)


def _members(groups: Sequence[str]) -> list[NDArray[np.intp]]:
    """Row indices of each group, groups in sorted order."""

    labels = np.asarray(groups)
    return [np.flatnonzero(labels == g).astype(np.intp) for g in sorted(set(groups))]
