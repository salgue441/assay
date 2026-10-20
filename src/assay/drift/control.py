"""EWMA control chart: track a smoothed series of per-epoch scores against baseline limits.

The exponentially weighted moving average reacts to small persistent shifts that a
Shewhart chart misses. Limits use the exact time-varying variance of the EWMA, so the
first points are not judged against the wider asymptotic band.
"""

import math
from dataclasses import dataclass

import numpy.typing as npt

from assay.drift._samples import as_sample


@dataclass(frozen=True)
class ChartPoint:
    """One observation on an EWMA chart with its smoothed value and control limits."""

    value: float
    ewma: float
    lower: float
    upper: float
    beyond_limits: bool


def ewma_chart(
    values: npt.ArrayLike,
    *,
    mean: float,
    sd: float,
    smoothing: float = 0.2,
    width: float = 3.0,
) -> tuple[ChartPoint, ...]:
    """Return the EWMA chart of ``values`` against an in-control ``mean`` and ``sd``.

    The EWMA starts at ``mean`` and updates as ``z_t = smoothing * x_t + (1 - smoothing) *
    z_(t-1)``. Point ``t`` (1-based) has limits ``mean +/- width * sd * sqrt(smoothing /
    (2 - smoothing) * (1 - (1 - smoothing) ** (2 * t)))``; a point is beyond the limits when
    its EWMA lies strictly outside them. An empty series gives an empty chart.

    Raises ``ValueError`` if ``sd <= 0``, ``smoothing`` is not in (0, 1], ``width <= 0``,
    ``mean`` is not finite, or ``values`` contains non-finite numbers.
    """
    if not math.isfinite(mean):
        raise ValueError(f"mean must be finite, got {mean}")
    if not sd > 0 or not math.isfinite(sd):
        raise ValueError(f"sd must be positive and finite, got {sd}")
    if not 0 < smoothing <= 1:
        raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
    if not width > 0:
        raise ValueError(f"width must be positive, got {width}")
    observations = as_sample(values, name="values", allow_empty=True)

    points: list[ChartPoint] = []
    ewma = mean
    for t, value in enumerate(observations.tolist(), start=1):
        ewma = smoothing * value + (1 - smoothing) * ewma
        half_width = width * sd * _ewma_sd_factor(smoothing, t)
        lower, upper = mean - half_width, mean + half_width
        points.append(
            ChartPoint(
                value=value,
                ewma=ewma,
                lower=lower,
                upper=upper,
                beyond_limits=ewma < lower or ewma > upper,
            )
        )
    return tuple(points)


def _ewma_sd_factor(smoothing: float, t: int) -> float:
    """Return the EWMA standard deviation at step ``t`` as a multiple of the process sd."""
    return math.sqrt(smoothing / (2 - smoothing) * (1 - (1 - smoothing) ** (2 * t)))
