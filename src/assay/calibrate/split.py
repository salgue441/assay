"""
Development/test splits that keep every group (repository) on one side, so
weights tuned on the development set are judged on unseen repositories.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Split[T]:
    dev: tuple[T, ...]
    test: tuple[T, ...]


def grouped_split[T](
    items: Sequence[T], *, group_of: Callable[[T], str], test_fraction: float, seed: int
) -> Split[T]:
    """
    Split `items` so that no group appears on both sides and the test side
    holds about `test_fraction` of the items.

    The sorted group names are shuffled with `seed`, then visited in that
    order: a group joins the test side when doing so brings the test item
    count closer to the target, unless it would leave the development side
    empty. If no group joined, the first shuffled group does. This greedy fill
    gets as close as whole groups allow in practice, though not always
    optimally. Items keep their input order on each side.
    """

    if not 0.0 < test_fraction < 1.0:
        msg = f"test_fraction must be in (0, 1), got {test_fraction}"
        raise ValueError(msg)
    sizes: dict[str, int] = {}
    for item in items:
        group = group_of(item)
        sizes[group] = sizes.get(group, 0) + 1
    if len(sizes) < 2:
        msg = f"a grouped split needs at least two groups, got {len(sizes)}"
        raise ValueError(msg)

    order = [sorted(sizes)[i] for i in np.random.default_rng(seed).permutation(len(sizes))]
    target = test_fraction * len(items)
    test_groups: set[str] = set()
    count = 0
    for group in order:
        grown = count + sizes[group]
        if grown < len(items) and abs(grown - target) < abs(count - target):
            test_groups.add(group)
            count = grown
    if not test_groups:
        test_groups.add(order[0])

    return Split(
        dev=tuple(item for item in items if group_of(item) not in test_groups),
        test=tuple(item for item in items if group_of(item) in test_groups),
    )
