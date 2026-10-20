"""
Development/test splits that keep every group (repository) on one side, so
weights tuned on the development set are judged on unseen repositories.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


MAX_EXHAUSTIVE_GROUPS = 12
"""Up to this many groups every assignment is tried; beyond, `grouped_split`."""


def stratified_grouped_split[T](
    items: Sequence[T],
    *,
    group_of: Callable[[T], str],
    strata_of: Callable[[T], Mapping[str, str]],
    test_fraction: float,
    seed: int,
) -> Split[T]:
    """
    Split `items` so that no group appears on both sides, the test side
    holds about `test_fraction` of the items, and each stratum variable
    (such as language) is distributed alike on both sides.

    Every assignment of groups to the test side is scored by its distance
    from the target size, as a share of all items, plus the total variation
    distance between the two sides' distributions of each stratum variable;
    the lowest score wins, ties going to the earlier assignment in an order
    shuffled with `seed`. Groups cannot be split, so balance is only as good
    as whole groups allow. With more than `MAX_EXHAUSTIVE_GROUPS` groups this
    falls back to the unstratified `grouped_split`.
    """

    if not 0.0 < test_fraction < 1.0:
        msg = f"test_fraction must be in (0, 1), got {test_fraction}"
        raise ValueError(msg)
    names = sorted({group_of(item) for item in items})
    if len(names) < 2:
        msg = f"a grouped split needs at least two groups, got {len(names)}"
        raise ValueError(msg)
    if len(names) > MAX_EXHAUSTIVE_GROUPS:
        return grouped_split(items, group_of=group_of, test_fraction=test_fraction, seed=seed)

    members: dict[str, list[Mapping[str, str]]] = {name: [] for name in names}
    for item in items:
        members[group_of(item)].append(strata_of(item))
    variables = sorted({v for strata in (s for m in members.values() for s in m) for v in strata})

    def score(test: frozenset[str]) -> float:
        test_rows = [s for g in names if g in test for s in members[g]]
        dev_rows = [s for g in names if g not in test for s in members[g]]
        size = abs(len(test_rows) - test_fraction * len(items)) / len(items)
        return size + sum(
            _total_variation(
                [row.get(v, "") for row in test_rows], [row.get(v, "") for row in dev_rows]
            )
            for v in variables
        )

    order = [names[i] for i in np.random.default_rng(seed).permutation(len(names))]
    candidates = [
        frozenset(name for bit, name in enumerate(order) if mask >> bit & 1)
        for mask in range(1, 2 ** len(order) - 1)
    ]
    test_groups = min(candidates, key=score)  # min keeps the first of equal scores
    return Split(
        dev=tuple(item for item in items if group_of(item) not in test_groups),
        test=tuple(item for item in items if group_of(item) in test_groups),
    )


def _total_variation(a: Sequence[str], b: Sequence[str]) -> float:
    """Half the L1 distance between the two value distributions, in [0, 1]."""

    values = set(a) | set(b)
    return 0.5 * sum(abs(a.count(v) / len(a) - b.count(v) / len(b)) for v in values)
