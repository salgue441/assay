import itertools
import math

import pytest

from assay.drift import ewma_chart


def test_ewma_starts_at_the_mean_and_matches_the_formula_at_t1() -> None:
    (point,) = ewma_chart([12.0], mean=10.0, sd=2.0, smoothing=0.2, width=3.0)

    half_width = 3.0 * 2.0 * math.sqrt(0.2 / 1.8 * (1 - 0.8**2))
    assert point.value == 12.0
    assert point.ewma == pytest.approx(0.2 * 12.0 + 0.8 * 10.0)
    assert point.lower == pytest.approx(10.0 - half_width)
    assert point.upper == pytest.approx(10.0 + half_width)
    # At t = 1 the EWMA sd equals smoothing * sd, so the band is width * smoothing * sd.
    assert point.upper - 10.0 == pytest.approx(3.0 * 0.2 * 2.0)
    assert not point.beyond_limits


def test_limits_widen_towards_the_asymptotic_band() -> None:
    chart = ewma_chart([0.0] * 60, mean=0.0, sd=1.0, smoothing=0.2, width=3.0)

    asymptote = 3.0 * math.sqrt(0.2 / 1.8)
    uppers = [point.upper for point in chart]
    assert all(a < b for a, b in itertools.pairwise(uppers))
    assert uppers[-1] == pytest.approx(asymptote)
    assert chart[-1].lower == pytest.approx(-asymptote)


def test_ewma_follows_the_recursion() -> None:
    chart = ewma_chart([1.0, 2.0, 3.0], mean=0.0, sd=1.0, smoothing=0.5)

    assert [point.ewma for point in chart] == pytest.approx([0.5, 1.25, 2.125])


def test_smoothing_of_one_is_a_shewhart_chart() -> None:
    chart = ewma_chart([0.5, 4.0], mean=0.0, sd=1.0, smoothing=1.0, width=3.0)

    assert [point.ewma for point in chart] == pytest.approx([0.5, 4.0])
    assert [point.upper for point in chart] == pytest.approx([3.0, 3.0])
    assert [point.beyond_limits for point in chart] == [False, True]


def test_injected_step_change_goes_beyond_limits() -> None:
    in_control = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0]
    shifted = [11.5] * 10

    chart = ewma_chart(in_control + shifted, mean=10.0, sd=1.0)

    assert not any(point.beyond_limits for point in chart[: len(in_control)])
    assert any(point.beyond_limits for point in chart[len(in_control) :])
    assert chart[-1].ewma > chart[-1].upper


def test_downward_step_goes_below_the_lower_limit() -> None:
    chart = ewma_chart([8.0] * 10, mean=10.0, sd=1.0)

    assert chart[-1].ewma < chart[-1].lower
    assert chart[-1].beyond_limits


def test_empty_series_gives_an_empty_chart() -> None:
    assert ewma_chart([], mean=0.0, sd=1.0) == ()


@pytest.mark.parametrize(
    ("mean", "sd", "smoothing", "width"),
    [
        (0.0, 0.0, 0.2, 3.0),
        (0.0, -1.0, 0.2, 3.0),
        (0.0, math.inf, 0.2, 3.0),
        (0.0, 1.0, 0.0, 3.0),
        (0.0, 1.0, 1.5, 3.0),
        (0.0, 1.0, 0.2, 0.0),
        (math.nan, 1.0, 0.2, 3.0),
    ],
)
def test_ewma_chart_rejects_invalid_parameters(
    mean: float, sd: float, smoothing: float, width: float
) -> None:
    with pytest.raises(ValueError):
        ewma_chart([1.0], mean=mean, sd=sd, smoothing=smoothing, width=width)
