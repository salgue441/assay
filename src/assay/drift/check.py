"""
The drift check: compare each later epoch's scores with the baseline epoch,
signal by signal, and apply the alert rules. Pure: callers pass the rows.

Two failures look alike on a dashboard: the subject's output getting worse,
and the judge or scorer itself shifting. Rolling-epoch signals watch the
first; golden-set agreement watches the second.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median, pstdev

from assay.drift.control import ewma_chart
from assay.drift.distribution import distribution_shift, population_stability_index
from assay.drift.rates import binomial_rate_test
from assay.drift.rules import consecutive_breaches, sustained_shift

QUALITY = "quality"
CRITICAL_RATE = "critical_rate"
METRIC = "metric:"
JUDGE_KAPPA = "judge_kappa:"
INPUT_SIZE = "input:unit_lines"
SUSTAINED_RUN = 8


@dataclass(frozen=True)
class EpochScore:
    """One scored document in one epoch."""

    document_id: str
    repo: str
    epoch: int
    composite: float
    rejected: bool
    components: Mapping[str, float]
    unit_lines: int | None
    """Source lines of the documented unit; None when its unit row is missing."""


@dataclass(frozen=True)
class Signal:
    name: str
    epoch: int | None
    """The epoch tested; None for signals of the whole run, like golden-set agreement."""
    statistic: float
    p_value: float | None
    breached: bool
    """This epoch's test failed."""
    alert: bool
    """The alert rule fired: for example, breaches in consecutive epochs."""
    context_only: bool
    """Reported with alerts but never alerting itself, like input shift."""
    detail: str


@dataclass(frozen=True)
class DriftCheck:
    baseline_epoch: int
    epochs: tuple[int, ...]
    """Epochs compared with the baseline, in order."""
    signals: tuple[Signal, ...]
    skipped: tuple[str, ...]
    """Signals that could not be tested, and why."""

    @property
    def alerts(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.alert)


def drift_check(
    scores: Sequence[EpochScore],
    *,
    ks_alpha: float,
    psi_floor: float,
    ewma_lambda: float,
    consecutive: int,
    golden_kappa: Mapping[str, float] | None = None,
    judge_kappa_floor: float = 0.55,
) -> DriftCheck:
    """
    Test every epoch after the earliest against the earliest (the baseline):

    - quality: KS and PSI of non-rejected composites; alerts after
      `consecutive` breaching epochs;
    - each metric: an EWMA chart of epoch means against the baseline mean,
      with the baseline spread scaled to the median epoch size; alerts when
      the EWMA leaves its limits or a run of eight epochs sits on one side;
    - critical-failure rate: an exact binomial test against the baseline
      rate; alerts when exceeded;
    - input size: PSI of unit line counts, as context only;
    - judge stability: `golden_kappa`, the judge's agreement with the frozen
      golden set per criterion; alerts below `judge_kappa_floor`.
    """

    by_epoch: dict[int, list[EpochScore]] = {}
    for score in scores:
        by_epoch.setdefault(score.epoch, []).append(score)
    signals: list[Signal] = []
    skipped: list[str] = []
    for criterion, kappa in sorted((golden_kappa or {}).items()):
        signals.append(
            Signal(
                name=f"{JUDGE_KAPPA}{criterion}",
                epoch=None,
                statistic=kappa,
                p_value=None,
                breached=kappa < judge_kappa_floor,
                alert=kappa < judge_kappa_floor,
                context_only=False,
                detail=f"golden-set kappa {kappa:.3f}, floor {judge_kappa_floor:g}",
            )
        )
    if golden_kappa is None:
        skipped.append("judge stability: no golden set is active")
    elif not golden_kappa:
        skipped.append("judge stability: kappa is undefined for every criterion")

    if not by_epoch:
        return DriftCheck(-1, (), tuple(signals), (*skipped, "no scores"))
    baseline_epoch, *later = sorted(by_epoch)
    baseline = by_epoch[baseline_epoch]
    if not later:
        skipped.append(f"epochs: only the baseline epoch {baseline_epoch} is scored")
        return DriftCheck(baseline_epoch, (), tuple(signals), tuple(skipped))

    epochs = [by_epoch[e] for e in later]
    signals += _quality(
        baseline,
        later,
        epochs,
        ks_alpha=ks_alpha,
        psi_floor=psi_floor,
        consecutive=consecutive,
        skipped=skipped,
    )
    signals += _critical_rate(baseline, later, epochs, ks_alpha)
    signals += _metrics(baseline, later, epochs, ewma_lambda, skipped)
    signals += _input_size(baseline, later, epochs, skipped)
    return DriftCheck(baseline_epoch, tuple(later), tuple(signals), tuple(skipped))


def _quality(
    baseline: Sequence[EpochScore],
    later: Sequence[int],
    epochs: Sequence[Sequence[EpochScore]],
    *,
    ks_alpha: float,
    psi_floor: float,
    consecutive: int,
    skipped: list[str],
) -> list[Signal]:
    reference = [s.composite for s in baseline if not s.rejected]
    if not reference:
        skipped.append("quality: every baseline document was rejected")
        return []
    signals: list[Signal] = []
    flags: list[bool] = []
    for epoch, scores in zip(later, epochs, strict=True):
        current = [s.composite for s in scores if not s.rejected]
        if not current:
            skipped.append(f"quality: every document of epoch {epoch} was rejected")
            flags.append(False)
            continue
        shift = distribution_shift(reference, current, ks_alpha=ks_alpha, psi_floor=psi_floor)
        flags.append(shift.breached)
        signals.append(
            Signal(
                name=QUALITY,
                epoch=epoch,
                statistic=shift.psi,
                p_value=shift.ks.p_value,
                breached=shift.breached,
                alert=consecutive_breaches(flags, required=consecutive),
                context_only=False,
                detail=(
                    f"mean composite {mean(current):.1f} vs {mean(reference):.1f}; "
                    f"KS {shift.ks.statistic:.3f}, PSI {shift.psi:.3f}"
                ),
            )
        )
    return signals


def _critical_rate(
    baseline: Sequence[EpochScore],
    later: Sequence[int],
    epochs: Sequence[Sequence[EpochScore]],
    alpha: float,
) -> list[Signal]:
    rate = sum(s.rejected for s in baseline) / len(baseline)
    signals = []
    for epoch, scores in zip(later, epochs, strict=True):
        failures = sum(s.rejected for s in scores)
        test = binomial_rate_test(failures, len(scores), baseline_rate=rate, alpha=alpha)
        signals.append(
            Signal(
                name=CRITICAL_RATE,
                epoch=epoch,
                statistic=test.rate,
                p_value=test.p_value,
                breached=test.exceeds,
                alert=test.exceeds,
                context_only=False,
                detail=f"{failures}/{len(scores)} rejected vs baseline rate {rate:.1%}",
            )
        )
    return signals


def _metrics(
    baseline: Sequence[EpochScore],
    later: Sequence[int],
    epochs: Sequence[Sequence[EpochScore]],
    smoothing: float,
    skipped: list[str],
) -> list[Signal]:
    names = sorted({n for s in baseline for n in s.components})
    typical_size = median(len(scores) for scores in epochs)
    signals = []
    for name in names:
        reference = [s.components[name] for s in baseline if name in s.components]
        series = [[s.components[name] for s in scores if name in s.components] for scores in epochs]
        if any(not values for values in series) or len(reference) < 2:
            skipped.append(f"{METRIC}{name}: missing in the baseline or in an epoch")
            continue
        spread = pstdev(reference) / math.sqrt(typical_size)
        if spread == 0:
            skipped.append(f"{METRIC}{name}: constant in the baseline")
            continue
        means = [mean(values) for values in series]
        center = mean(reference)
        chart = ewma_chart(means, mean=center, sd=spread, smoothing=smoothing)
        for i, (epoch, point) in enumerate(zip(later, chart, strict=True)):
            sustained = sustained_shift(means[: i + 1], mean=center, run_length=SUSTAINED_RUN)
            signals.append(
                Signal(
                    name=f"{METRIC}{name}",
                    epoch=epoch,
                    statistic=point.ewma,
                    p_value=None,
                    breached=point.beyond_limits,
                    alert=point.beyond_limits or sustained,
                    context_only=False,
                    detail=(
                        f"EWMA {point.ewma:.3f} in [{point.lower:.3f}, {point.upper:.3f}]"
                        + ("; sustained shift" if sustained else "")
                    ),
                )
            )
    return signals


def _input_size(
    baseline: Sequence[EpochScore],
    later: Sequence[int],
    epochs: Sequence[Sequence[EpochScore]],
    skipped: list[str],
) -> list[Signal]:
    reference = [s.unit_lines for s in baseline if s.unit_lines is not None]
    signals = []
    for epoch, scores in zip(later, epochs, strict=True):
        current = [s.unit_lines for s in scores if s.unit_lines is not None]
        if not reference or not current:
            skipped.append(f"{INPUT_SIZE} @ epoch {epoch}: no unit rows to size")
            continue
        psi = population_stability_index(reference, current)
        signals.append(
            Signal(
                name=INPUT_SIZE,
                epoch=epoch,
                statistic=psi,
                p_value=None,
                breached=False,
                alert=False,
                context_only=True,
                detail=f"PSI of unit line counts {psi:.3f}",
            )
        )
    return signals
