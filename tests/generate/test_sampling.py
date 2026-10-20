from collections import Counter

import pytest

from assay.generate import sample_units

from .units import units

UNITS = units()


def kinds(sample: list) -> list[str]:  # type: ignore[type-arg]
    return [u.kind.value for u in sample]


def test_kinds_take_turns() -> None:
    sample = sample_units(UNITS, 3, seed=1)

    assert sorted(kinds(sample)) == ["class", "function", "module"]


def test_a_kind_that_runs_out_leaves_the_rest_to_others() -> None:
    sample = sample_units(UNITS, None, seed=1)

    assert Counter(kinds(sample)) == Counter(kinds(UNITS))
    assert kinds(sample)[-2:] in (["class", "function"], ["function", "class"])


def test_the_same_seed_gives_the_same_sample() -> None:
    assert sample_units(UNITS, 3, seed=7) == sample_units(list(reversed(UNITS)), 3, seed=7)


def test_the_choice_follows_units_across_epochs() -> None:
    later = units(commit="d" * 40)

    first = [u.unit_id for u in sample_units(UNITS, 4, seed=7)]
    again = [u.unit_id for u in sample_units(later, 4, seed=7)]

    assert first == again


def test_size_larger_than_the_population_returns_everything() -> None:
    assert len(sample_units(UNITS, 100, seed=1)) == len(UNITS)


def test_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sample_units(UNITS, 0, seed=1)
