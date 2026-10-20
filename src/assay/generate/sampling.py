"""
Choose which units get documented.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain, zip_longest

from assay.corpus import Unit


def sample_units(units: Sequence[Unit], size: int | None, *, seed: int) -> list[Unit]:
    """
    A deterministic sample balanced across unit kinds, in a stable order.

    Within each kind, units are ordered by a seeded hash of their `unit_id`,
    and kinds take turns, so modules, classes and functions are represented
    equally as far as each kind allows. Because the order depends only on
    `unit_id`, the same units are chosen again in later epochs wherever they
    still exist. `size=None` returns every unit in that order.
    """

    if size is not None and size < 1:
        msg = f"sample size must be at least 1, got {size}"
        raise ValueError(msg)

    by_kind: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        by_kind[unit.kind.value].append(unit)

    def rank(unit: Unit) -> str:
        return hashlib.sha256(f"{seed}:{unit.unit_id}".encode()).hexdigest()

    queues = [sorted(group, key=rank) for _, group in sorted(by_kind.items())]
    interleaved = [u for u in chain.from_iterable(zip_longest(*queues)) if u is not None]
    return interleaved if size is None else interleaved[:size]
