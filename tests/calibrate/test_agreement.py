import pytest

from assay.calibrate import krippendorff_alpha, weighted_kappa, weighted_kappa_estimate

# Observed: (1,1), (2,2), (3,2). Rater a's marginals are (1/3, 1/3, 1/3) and
# rater b's (1/3, 2/3, 0). With |i - j| weights the observed disagreement is
# 1/3 and the chance disagreement 7/9, so kappa = 1 - 3/7 = 4/7. With
# (i - j)^2 weights they are 1/3 and 1, so kappa = 2/3.
A = [1, 2, 3]
B = [1, 2, 2]

RATER_1 = [5, 4, 4, 3, 2, 1, 5, 3, 2, 4]
RATER_2 = [5, 4, 3, 3, 2, 2, 4, 3, 1, 4]
REPOS = ["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4", "r5", "r5"]


@pytest.mark.parametrize(("weights", "expected"), [("linear", 4 / 7), ("quadratic", 2 / 3)])
def test_kappa_matches_a_hand_computed_example(weights: str, expected: float) -> None:
    kappa = weighted_kappa(A, B, labels=(1, 2, 3), weights=weights)  # type: ignore[arg-type]

    assert kappa == pytest.approx(expected)


def test_identical_ratings_give_kappa_one() -> None:
    assert weighted_kappa(RATER_1, RATER_1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ([1, 2], [1], "equally long"),
        ([], [], "at least one"),
        ([1, 6], [1, 2], "outside the label set"),
        ([3, 3], [3, 3], "undefined"),
    ],
)
def test_invalid_kappa_inputs_raise(a: list[int], b: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_kappa(a, b)


def test_the_kappa_estimate_wraps_the_point_value_reproducibly() -> None:
    first = weighted_kappa_estimate(RATER_1, RATER_2, groups=REPOS, resamples=300, seed=2)
    again = weighted_kappa_estimate(RATER_1, RATER_2, groups=REPOS, resamples=300, seed=2)

    assert first == again
    assert first.value == pytest.approx(weighted_kappa(RATER_1, RATER_2))
    assert first.ci_low <= first.value <= first.ci_high
    assert first.n == 10


def test_alpha_matches_a_hand_computed_example() -> None:
    # Coincidences o11 = 2, o12 = o21 = 1, o22 = 4, so n1 = 3, n2 = 5, n = 8.
    # Do = 2/8 and De = 2 * 3 * 5 / (8 * 7) = 30/56, so alpha = 1 - 14/30.
    # Item i5 has a single rating and is not pairable.
    ratings = {
        "i1": {"ann": 1, "bob": 1},
        "i2": {"ann": 1, "bob": 2},
        "i3": {"ann": 2, "bob": 2},
        "i4": {"ann": 2, "bob": 2},
        "i5": {"ann": 1, "bob": None},
    }

    assert krippendorff_alpha(ratings, level="nominal") == pytest.approx(8 / 15)


def test_perfect_agreement_gives_alpha_one() -> None:
    ratings = {
        "i1": {"ann": 5, "bob": 5, "cy": 5},
        "i2": {"ann": 2, "bob": 2},
        "i3": {"bob": 3, "cy": 3},
        "i4": {"ann": 1, "cy": 1},
    }

    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("ratings", "message"),
    [
        ({"i1": {"ann": 1}, "i2": {"ann": 2, "bob": None}}, "two raters"),
        ({"i1": {"ann": 1}, "i2": {"bob": 2}}, "two or more raters"),
        ({"i1": {"ann": 3, "bob": 3}, "i2": {"ann": 3, "bob": 3}}, "same value"),
    ],
)
def test_invalid_alpha_inputs_raise(
    ratings: dict[str, dict[str, int | None]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        krippendorff_alpha(ratings)
