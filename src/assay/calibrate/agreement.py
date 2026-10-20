"""
Agreement between raters: weighted Cohen's kappa for a pair of raters and
Krippendorff's alpha for any number of raters with missing ratings.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import krippendorff
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

from assay.calibrate.bootstrap import Estimate, bootstrap_interval

KappaWeights = Literal["linear", "quadratic"]
AlphaLevel = Literal["nominal", "ordinal", "interval"]

LIKERT = (1, 2, 3, 4, 5)


def weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    labels: Sequence[int] = LIKERT,
    weights: KappaWeights = "quadratic",
) -> float:
    """
    Weighted Cohen's kappa between two raters who rated the same items.
    Raises ValueError when kappa is undefined, i.e. when chance agreement is
    already perfect (both raters used one and the same label throughout).
    """

    left, right = _paired(a, b, labels)
    value = _kappa(left, right, labels, weights)
    if np.isnan(value):
        msg = "kappa is undefined when both raters use a single, shared label"
        raise ValueError(msg)
    return value


def weighted_kappa_estimate(
    a: Sequence[int],
    b: Sequence[int],
    *,
    labels: Sequence[int] = LIKERT,
    weights: KappaWeights = "quadratic",
    groups: Sequence[str] | None = None,
    resamples: int = 10_000,
    seed: int,
    confidence: float = 0.95,
) -> Estimate:
    """
    `weighted_kappa` with a percentile bootstrap interval; given `groups`,
    the bootstrap resamples whole groups.
    """

    value = weighted_kappa(a, b, labels=labels, weights=weights)
    left, right = _paired(a, b, labels)

    def statistic(rows: NDArray[np.intp]) -> float:
        return _kappa(left[rows], right[rows], labels, weights)

    low, high = bootstrap_interval(
        statistic, len(left), groups=groups, resamples=resamples, seed=seed, confidence=confidence
    )
    return Estimate(value=value, ci_low=low, ci_high=high, n=len(left), confidence=confidence)


def krippendorff_alpha(
    ratings: Mapping[str, Mapping[str, int | None]], *, level: AlphaLevel = "ordinal"
) -> float:
    """
    Krippendorff's alpha over `ratings`, which maps item -> rater -> value;
    a rater who did not rate an item is absent or None. The value domain is
    the set of values that occur, so for the ordinal level an unused point
    on the scale does not count.
    """

    raters = sorted(
        {r for by_rater in ratings.values() for r, v in by_rater.items() if v is not None}
    )
    if len(raters) < 2:
        msg = f"alpha needs at least two raters, got {len(raters)}"
        raise ValueError(msg)
    items = sorted(ratings)
    matrix = np.array(
        [[_value(ratings[item].get(rater)) for item in items] for rater in raters],
        dtype=np.float64,
    )
    rated = np.sum(~np.isnan(matrix), axis=0)
    if not np.any(rated >= 2):
        msg = "alpha needs at least one item rated by two or more raters"
        raise ValueError(msg)
    pairable = matrix[:, rated >= 2]
    if len(np.unique(pairable[~np.isnan(pairable)])) < 2:
        msg = "alpha is undefined when every pairable rating has the same value"
        raise ValueError(msg)
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))


def _paired(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        msg = f"a and b must be 1-D and equally long, got {left.shape} and {right.shape}"
        raise ValueError(msg)
    if len(left) == 0:
        msg = "kappa needs at least one pair of ratings"
        raise ValueError(msg)
    unknown = sorted(set(np.concatenate([left, right]).tolist()) - set(labels))
    if unknown:
        msg = f"ratings outside the label set {tuple(labels)}: {unknown}"
        raise ValueError(msg)
    return left, right


def _kappa(
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    labels: Sequence[int],
    weights: KappaWeights,
) -> float:
    """Kappa, or NaN when it is undefined."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # undefined kappa warns, then gives NaN
        return float(cohen_kappa_score(left, right, labels=list(labels), weights=weights))


def _value(rating: int | None) -> float:
    return np.nan if rating is None else float(rating)
