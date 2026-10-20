"""Run rules: decide from recent history whether a shift is sustained or an alert is due."""

from collections.abc import Sequence

import numpy.typing as npt

from assay.drift._samples import as_sample


def sustained_shift(values: npt.ArrayLike, *, mean: float, run_length: int = 8) -> bool:
    """Return whether the last ``run_length`` values all lie strictly on one side of ``mean``.

    This is the classic "run of eight" rule: a long run above or below the centre line is
    unlikely by chance and signals a persistent shift. A value equal to ``mean`` breaks the
    run. Fewer than ``run_length`` values never form a run.

    Raises ``ValueError`` if ``run_length < 1`` or ``values`` contains non-finite numbers.
    """
    if run_length < 1:
        raise ValueError(f"run_length must be at least 1, got {run_length}")
    observations = as_sample(values, name="values", allow_empty=True)
    if observations.size < run_length:
        return False
    recent = observations[-run_length:]
    return bool((recent > mean).all() or (recent < mean).all())


def consecutive_breaches(flags: Sequence[bool], *, required: int) -> bool:
    """Return whether the last ``required`` flags are all true.

    Alerts fire only after consecutive breaching runs, so one noisy run cannot raise one.
    Fewer than ``required`` flags never qualify.

    Raises ``ValueError`` if ``required < 1``.
    """
    if required < 1:
        raise ValueError(f"required must be at least 1, got {required}")
    if len(flags) < required:
        return False
    return all(flags[-required:])
