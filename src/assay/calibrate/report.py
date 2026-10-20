"""
Render a calibration report as text for the terminal and as JSON to keep.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping

from assay.calibrate.ablation import AblationStudy
from assay.calibrate.bootstrap import Estimate
from assay.calibrate.study import CalibrationReport

_SIDES = {"dev": "development", "test": "test"}


def report_json(report: CalibrationReport) -> str:
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True) + "\n"


def report_text(report: CalibrationReport, *, rho_target: float, kappa_target: float) -> str:
    lines = [
        f"{report.documents} rated documents from raters {', '.join(report.raters)}",
        f"development: {_repos(report.dev_repos)} ({report.dev_documents} documents)",
        f"test: {_repos(report.test_repos)} ({report.test_documents} documents)"
        + ("" if "test" in report.correlation else ", held out"),
    ]

    if report.rater_alpha:
        lines.append("")
        lines.append("Rater agreement (Krippendorff's alpha, ordinal)")
        lines += [f"  {name:<16} {value:6.3f}" for name, value in report.rater_alpha.items()]

    for side, label in _SIDES.items():
        if side not in report.correlation:
            continue
        lines.append("")
        lines.append(f"{label.capitalize()} split")
        rho = rho_target if side == "test" else None
        kappa = kappa_target if side == "test" else None
        lines += _estimates("composite vs overall", report.correlation[side], rho)
        if side in report.fitted_correlation:
            lines += _estimates("fitted vs overall", report.fitted_correlation[side], rho)
        lines += _estimates("judge vs human kappa", report.judge_kappa[side], kappa)
        lines.append("  decisions (scorer -> human)")
        for scorer, humans in report.decisions[side].items():
            counts = ", ".join(f"{human} {count}" for human, count in humans.items())
            lines.append(f"    {scorer:<9} {counts}")
        lines += _ablations(report.ablations[side])

    if report.fitted is not None:
        lines.append("")
        lines.append(
            f"Fitted weights (shrinkage {report.fitted.shrinkage:g}, "
            f"{report.fitted.documents} documents)"
        )
        lines += [f"  {name:<24} {w:.3f}" for name, w in sorted(report.fitted.weights.items())]

    if report.disagreements:
        lines.append("")
        lines.append("Largest rank gaps on development (score percentile - human percentile)")
        lines += [
            f"  {g.rank_gap:+.2f}  composite {g.composite:5.1f}  human {g.overall:.1f}  "
            f"{g.unit_id}  ({g.subject})"
            for g in report.disagreements
        ]

    if report.notes:
        lines.append("")
        lines.append("Not computed")
        lines += [f"  {note}" for note in report.notes]
    return "\n".join(lines)


def _estimates(title: str, estimates: Mapping[str, Estimate], target: float | None) -> list[str]:
    """Targets apply to the held-out test split, so only its lines show a verdict."""

    lines = [f"  {title}"]
    for name, e in estimates.items():
        line = f"    {name:<14} {e.value:6.3f}  [{e.ci_low:6.3f}, {e.ci_high:6.3f}]  n={e.n}"
        if target is not None:
            line += f"  {'meets' if e.value >= target else 'below'} target {target:g}"
        lines.append(line)
    return lines


def _ablations(study: AblationStudy) -> list[str]:
    full = study.full
    labels = ["full scorer", *(f"without {a.removed}" for a in study.ablations)]
    width = max(map(len, labels))
    lines = [
        "  ablation (prior weights; change from the full scorer)",
        f"    {labels[0]:<{width}}  rho {_number(full.spearman)}  "
        f"decisions {_number(full.decision_agreement)}  n={full.documents}",
    ]
    lines += [
        f"    {label:<{width}}  rho {_number(a.delta_spearman, signed=True)}  "
        f"decisions {_number(a.delta_decision_agreement, signed=True)}  n={a.outcome.documents}"
        for label, a in zip(labels[1:], study.ablations, strict=True)
    ]
    return lines


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "   n/a"
    return f"{value:+6.3f}" if signed else f"{value:6.3f}"


def _repos(repos: tuple[str, ...]) -> str:
    return ", ".join(repos) if repos else "none"
