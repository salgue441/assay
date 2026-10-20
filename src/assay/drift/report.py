"""
Render a drift check for the terminal, and as rows for the `drift_runs` log.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from assay.drift.check import DriftCheck, Signal


def drift_rows(
    check: DriftCheck, *, run_at: datetime, scorer_version: str, golden_version: str | None
) -> list[dict[str, Any]]:
    """One row per signal, with what identifies the run: a `drift_runs` log."""

    run_id = run_at.strftime("%Y%m%dT%H%M%S%fZ")
    return [
        {
            "run_id": run_id,
            "run_at": run_at.isoformat(),
            "scorer_version": scorer_version,
            "golden_version": golden_version,
            "baseline_epoch": check.baseline_epoch,
            "signal": s.name,
            "epoch": s.epoch,
            "statistic": s.statistic,
            "p_value": s.p_value,
            "breached": s.breached,
            "alert": s.alert,
            "context_only": s.context_only,
            "detail": s.detail,
        }
        for s in check.signals
    ]


def drift_text(check: DriftCheck) -> str:
    lines = []
    if check.epochs:
        lines.append(
            f"Drift of epochs {', '.join(map(str, check.epochs))} "
            f"against baseline epoch {check.baseline_epoch}"
        )
    alerts = check.alerts
    lines.append(f"{len(alerts)} alert{'s' if len(alerts) != 1 else ''}")
    lines += [f"  ALERT {_label(s)}  {s.detail}" for s in alerts]

    latest = max(check.epochs) if check.epochs else None
    current = [s for s in check.signals if not s.alert and s.epoch in (latest, None)]
    if current:
        lines.append("")
        lines.append("Latest results" + (f" (epoch {latest})" if latest is not None else ""))
        lines += [
            f"  {'breach' if s.breached else 'ok':<6} {_label(s)}  {s.detail}"
            + ("  (context)" if s.context_only else "")
            for s in current
        ]
    if check.skipped:
        lines.append("")
        lines.append("Not tested")
        lines += [f"  {reason}" for reason in check.skipped]
    return "\n".join(lines)


def _label(signal: Signal) -> str:
    where = "" if signal.epoch is None else f" @ epoch {signal.epoch}"
    return f"{signal.name}{where}"
