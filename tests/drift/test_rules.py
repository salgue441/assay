import pytest

from assay.drift import consecutive_breaches, sustained_shift


def test_run_above_the_mean_is_a_sustained_shift() -> None:
    assert sustained_shift([0.0, -1.0] + [0.5] * 8, mean=0.0)


def test_run_below_the_mean_is_a_sustained_shift() -> None:
    assert sustained_shift([1.0] + [-0.5] * 8, mean=0.0)


def test_mixed_sides_are_not_a_sustained_shift() -> None:
    assert not sustained_shift([0.5] * 7 + [-0.5], mean=0.0)


def test_value_on_the_mean_breaks_the_run() -> None:
    assert not sustained_shift([0.5] * 7 + [0.0], mean=0.0)


def test_too_few_values_are_not_a_sustained_shift() -> None:
    assert not sustained_shift([0.5] * 7, mean=0.0)
    assert not sustained_shift([], mean=0.0)


def test_run_length_is_configurable() -> None:
    assert sustained_shift([-1.0, 2.0, 3.0], mean=0.0, run_length=2)


def test_sustained_shift_rejects_invalid_run_length() -> None:
    with pytest.raises(ValueError, match="run_length"):
        sustained_shift([1.0], mean=0.0, run_length=0)


@pytest.mark.parametrize(
    ("flags", "required", "expected"),
    [
        ([True, True], 2, True),
        ([False, True, True], 2, True),
        ([True, True, False], 2, False),
        ([True, False, True], 2, False),
        ([True], 2, False),
        ([], 1, False),
        ([True], 1, True),
        ([False], 1, False),
    ],
)
def test_consecutive_breaches(flags: list[bool], required: int, expected: bool) -> None:
    assert consecutive_breaches(flags, required=required) is expected


def test_consecutive_breaches_rejects_invalid_required() -> None:
    with pytest.raises(ValueError, match="required"):
        consecutive_breaches([True], required=0)
