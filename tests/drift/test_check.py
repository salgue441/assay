import re
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from assay.drift.check import (
    CRITICAL_RATE,
    INPUT_SIZE,
    QUALITY,
    DriftCheck,
    EpochScore,
    Signal,
    drift_check,
)
from assay.drift.report import drift_rows, drift_text

N = 60


def epoch(
    number: int,
    *,
    center: float = 70.0,
    rejected: float = 0.0,
    lines: int = 20,
    seed: int | None = None,
) -> list[EpochScore]:
    rng = np.random.default_rng(number if seed is None else seed)
    composites = np.clip(rng.normal(center, 8.0, size=N), 0, 100)
    return [
        EpochScore(
            document_id=f"{number}-{i}",
            repo="o/r",
            epoch=number,
            composite=float(c),
            rejected=i < rejected * N,
            components={"readability": float(c) / 100},
            unit_lines=lines + i % 7,
        )
        for i, c in enumerate(composites)
    ]


def check(*epochs: list[EpochScore], golden_kappa: dict[str, float] | None = None) -> DriftCheck:
    return drift_check(
        [s for e in epochs for s in e],
        ks_alpha=0.01,
        psi_floor=0.2,
        ewma_lambda=0.2,
        consecutive=2,
        golden_kappa=golden_kappa,
    )


def signals(result: DriftCheck, name: str) -> dict[int | None, Signal]:
    return {s.epoch: s for s in result.signals if s.name == name}


def test_stable_epochs_raise_no_alert() -> None:
    result = check(epoch(0), epoch(1), epoch(2), epoch(3))

    assert result.baseline_epoch == 0
    assert result.epochs == (1, 2, 3)
    assert result.alerts == ()
    assert set(signals(result, QUALITY)) == {1, 2, 3}


def test_a_quality_drop_alerts_only_after_consecutive_breaches() -> None:
    result = check(epoch(0), epoch(1, center=45), epoch(2, center=45))

    quality = signals(result, QUALITY)
    assert quality[1].breached and not quality[1].alert
    assert quality[2].breached and quality[2].alert


def test_a_one_epoch_dip_does_not_alert() -> None:
    result = check(epoch(0), epoch(1, center=45), epoch(2))

    assert not any(s.alert for s in signals(result, QUALITY).values())


def test_a_rise_in_rejections_alerts() -> None:
    result = check(epoch(0, rejected=0.05), epoch(1, rejected=0.4))

    assert signals(result, CRITICAL_RATE)[1].alert


def test_a_metric_shift_leaves_the_control_limits() -> None:
    result = check(epoch(0), epoch(1, center=50))

    metric = signals(result, "metric:readability")[1]
    assert metric.breached and metric.alert


def test_input_shift_is_context_and_never_alerts() -> None:
    result = check(epoch(0), epoch(1, lines=400))

    size = signals(result, INPUT_SIZE)[1]
    assert size.context_only
    assert size.statistic > 1
    assert not size.alert


def test_the_judge_is_checked_against_the_golden_set() -> None:
    result = check(epoch(0), golden_kappa={"accuracy": 0.3, "clarity": 0.8})

    judged = {s.name: s for s in result.signals if s.epoch is None}
    assert judged["judge_kappa:accuracy"].alert
    assert not judged["judge_kappa:clarity"].alert


@pytest.mark.parametrize(
    ("kappa", "reason"),
    [(None, "no golden set is active"), ({}, "undefined for every criterion")],
)
def test_missing_judge_stability_is_explained(kappa: dict[str, float] | None, reason: str) -> None:
    result = check(epoch(0), epoch(1), golden_kappa=kappa)

    assert any(reason in note for note in result.skipped)


def test_a_baseline_alone_has_nothing_to_compare() -> None:
    result = check(epoch(0))

    assert result.epochs == ()
    assert any("only the baseline epoch 0" in note for note in result.skipped)


def test_no_scores_at_all() -> None:
    assert "no scores" in check().skipped


def test_each_signal_becomes_a_logged_row() -> None:
    result = check(epoch(0), epoch(1, center=45), epoch(2, center=45))
    run_at = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

    rows = drift_rows(result, run_at=run_at, scorer_version="s" * 16, golden_version=None)

    assert len(rows) == len(result.signals)
    assert {r["run_id"] for r in rows} == {"20261020T120000000000Z"}
    assert sum(r["alert"] for r in rows) == len(result.alerts)


def test_the_text_leads_with_alerts() -> None:
    text = drift_text(check(epoch(0), epoch(1, center=45), epoch(2, center=45)))

    assert re.fullmatch(r"\d+ alerts?", text.splitlines()[1])
    assert "ALERT quality @ epoch 2" in text


def test_documents_without_unit_rows_are_left_out_of_input_size() -> None:
    unsized = [replace(s, unit_lines=None) for s in epoch(1)]

    result = check(epoch(0), unsized)

    assert INPUT_SIZE not in {s.name for s in result.signals}
    assert any("no unit rows to size" in note for note in result.skipped)
