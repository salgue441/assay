"""
Composite weights fitted to human ratings, shrunk toward a prior.

The composite score is a convex combination of component scores, so the
weights are non-negative and sum to one. With few rated documents a free fit
overfits, so the fit is pulled toward the prior weights and the strength of
that pull is chosen by cross-validation over held-out groups (repositories).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]

SIMPLEX_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12
"""Held-out errors this close (absolutely, or relative 1e-9) count as a tie."""


def fit_weights(
    features: FloatArray, target: FloatArray, *, prior: FloatArray, shrinkage: float
) -> FloatArray:
    """
    The weights `w` minimising
    `||target - features @ w||^2 + shrinkage * ||w - prior||^2` subject to
    `w >= 0` and `sum(w) == 1`, found with SLSQP starting from the prior.
    `features` is n x k, `target` has n entries and `prior` k, on the simplex.
    """

    x, y, p = _validated(features, target, prior)
    if not (math.isfinite(shrinkage) and shrinkage >= 0):
        msg = f"shrinkage must be a finite number >= 0, got {shrinkage}"
        raise ValueError(msg)

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        residual = y - x @ w
        offset = w - p
        value = float(residual @ residual + shrinkage * (offset @ offset))
        gradient = -2.0 * (x.T @ residual) + 2.0 * shrinkage * offset
        return value, gradient

    result = minimize(
        objective,
        p,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(p),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": np.ones_like}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        msg = f"weight fit did not converge: {result.message}"
        raise RuntimeError(msg)
    weights: FloatArray = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    normalised: FloatArray = weights / weights.sum()
    return normalised


def choose_shrinkage(
    features: FloatArray,
    target: FloatArray,
    *,
    groups: Sequence[str],
    prior: FloatArray,
    candidates: Sequence[float],
    folds: int = 5,
    seed: int,
) -> float:
    """
    The candidate shrinkage with the lowest held-out mean squared error under
    grouped cross-validation (see `grouped_folds`); the error is pooled over
    every held-out row. Ties go to the larger shrinkage, the more
    conservative choice.
    """

    x, y, p = _validated(features, target, prior)
    if len(groups) != len(y):
        msg = f"groups has {len(groups)} entries for {len(y)} rows"
        raise ValueError(msg)
    if not candidates:
        msg = "choose_shrinkage needs at least one candidate"
        raise ValueError(msg)

    held_out = grouped_folds(groups, folds=folds, seed=seed)
    errors: dict[float, float] = {}
    for shrinkage in sorted(set(candidates), reverse=True):
        squared = 0.0
        for test in held_out:
            train = np.setdiff1d(np.arange(len(y)), test)
            w = fit_weights(x[train], y[train], prior=p, shrinkage=shrinkage)
            residual = y[test] - x[test] @ w
            squared += float(residual @ residual)
        errors[shrinkage] = squared / len(y)

    best = min(errors.values())
    return next(
        s
        for s, error in errors.items()
        if math.isclose(error, best, rel_tol=1e-9, abs_tol=TIE_TOLERANCE)
    )


def grouped_folds(groups: Sequence[str], *, folds: int, seed: int) -> list[NDArray[np.intp]]:
    """
    Held-out row indices for each of `folds` folds; every group falls in
    exactly one fold. The sorted group names are shuffled with `seed`, then
    each joins the fold with the fewest rows so far (lowest index on a tie).
    """

    names = sorted(set(groups))
    if not 2 <= folds <= len(names):
        msg = f"folds must be between 2 and the number of groups ({len(names)}), got {folds}"
        raise ValueError(msg)
    labels = np.asarray(groups)
    assigned: list[list[NDArray[np.intp]]] = [[] for _ in range(folds)]
    counts = [0] * folds
    for i in np.random.default_rng(seed).permutation(len(names)):
        rows = np.flatnonzero(labels == names[i]).astype(np.intp)
        fold = counts.index(min(counts))
        assigned[fold].append(rows)
        counts[fold] += len(rows)
    return [np.sort(np.concatenate(parts)) for parts in assigned]


def _validated(
    features: FloatArray, target: FloatArray, prior: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or p.ndim != 1:
        msg = f"expected features n x k, target n, prior k; got {x.shape}, {y.shape}, {p.shape}"
        raise ValueError(msg)
    if x.shape != (len(y), len(p)):
        msg = f"features {x.shape} do not match target {y.shape} and prior {p.shape}"
        raise ValueError(msg)
    if np.any(p < 0) or not math.isclose(p.sum(), 1.0, abs_tol=SIMPLEX_TOLERANCE):
        msg = f"prior must be non-negative and sum to 1, got {p.tolist()}"
        raise ValueError(msg)
    return x, y, p
