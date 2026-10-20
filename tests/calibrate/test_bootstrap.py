from collections import Counter

import numpy as np
import pytest
from numpy.typing import NDArray

from assay.calibrate import bootstrap_interval, resample_indices

GROUPS = ["a", "a", "a", "b", "b", "c", "d", "d", "d", "d"]


def test_an_ungrouped_replicate_draws_n_rows() -> None:
    rows = resample_indices(5, None, np.random.default_rng(0))

    assert len(rows) == 5
    assert set(rows.tolist()) <= set(range(5))


def test_a_clustered_replicate_never_splits_a_group() -> None:
    rng = np.random.default_rng(7)
    groups = {g: [i for i, label in enumerate(GROUPS) if label == g] for g in set(GROUPS)}

    for _ in range(200):
        counts = Counter(resample_indices(len(GROUPS), GROUPS, rng).tolist())
        for rows in groups.values():
            assert len({counts[i] for i in rows}) == 1


def test_undefined_replicates_are_skipped() -> None:
    calls = iter(range(100))

    def statistic(rows: NDArray[np.intp]) -> float:
        return float("nan") if next(calls) % 4 == 0 else 1.0

    interval = bootstrap_interval(statistic, 3, groups=None, resamples=100, seed=0, confidence=0.9)

    assert interval == (1.0, 1.0)


def test_too_few_defined_replicates_raise() -> None:
    with pytest.raises(ValueError, match="replicates were defined"):
        bootstrap_interval(
            lambda rows: float("nan"), 3, groups=None, resamples=10, seed=0, confidence=0.9
        )


@pytest.mark.parametrize(
    ("resamples", "confidence", "groups"),
    [(0, 0.95, None), (10, 1.0, None), (10, 0.0, None), (10, 0.95, ["a"])],
)
def test_invalid_options_raise(resamples: int, confidence: float, groups: list[str] | None) -> None:
    with pytest.raises(ValueError):
        bootstrap_interval(
            lambda rows: 0.0,
            3,
            groups=groups,
            resamples=resamples,
            seed=0,
            confidence=confidence,
        )
