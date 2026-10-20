"""
Render a calibration report as text for the terminal and as JSON to keep.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

from assay.calibrate.ablation import AblationStudy
from assay.calibrate.bias import BiasStudy
from assay.calibrate.bootstrap import Estimate
from assay.calibrate.study import CalibrationReport
from assay.calibrate.variance import VarianceStudy

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
        lines += _confusion("decisions (scorer -> human)", report.decisions[side])
        if side in report.chosen_decisions:
            lines += _confusion("decisions with chosen thresholds", report.chosen_decisions[side])
        lines += _decision_quality(report, side)
        lines += _ablations(report.ablations[side])
        lines += _estimates(
            "predictors vs overall (spearman; models out-of-fold on development)",
            report.baseline[side],
            rho,
        )

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


def _decision_quality(report: CalibrationReport, side: str) -> list[str]:
    lines = ["  decision quality (undisputed human decisions)"]
    for name, quality in report.decision_quality[side].items():
        routing = report.chosen_routing if name == "chosen" else None
        thresholds = (
            f" (publish >= {routing.publish_at_or_above:g}, escalate < {routing.escalate_below:g})"
            if routing
            else ""
        )
        lines.append(
            f"    {name + thresholds:<46} agreement {_number(quality.agreement)}  "
            f"publish precision {_number(quality.publish_precision)} "
            f"of {quality.published}  n={quality.decided}"
        )
    return lines


def _confusion(title: str, table: Mapping[str, Mapping[str, int]]) -> list[str]:
    lines = [f"  {title}"]
    for scorer, humans in table.items():
        counts = ", ".join(f"{human} {count}" for human, count in humans.items())
        lines.append(f"    {scorer:<9} {counts}")
    return lines


def variance_json(studies: Sequence[VarianceStudy]) -> str:
    return json.dumps([dataclasses.asdict(s) for s in studies], indent=2, sort_keys=True) + "\n"


def variance_text(studies: Sequence[VarianceStudy], *, stability_target: float) -> str:
    lines: list[str] = []
    for study in studies:
        if lines:
            lines.append("")
        lines.append(f"Temperature {study.temperature:g}, up to {study.runs} runs per criterion")
        lines.append("  grade spread per document (1-5 scale)")
        lines += [
            f"    {s.criterion:<14} mean sd {s.mean_sd:5.2f}  max sd {s.max_sd:5.2f}  "
            f"changed in {s.changed:6.1%} of {s.documents}"
            for s in study.criteria
        ]
        if study.band_stability is None:
            lines.append("  routing stability: no document has two complete runs")
            continue
        verdict = "meets" if study.band_stability >= stability_target else "below"
        lines.append(
            f"  routing stability: {study.band_stability:.1%} of {study.documents} documents "
            f"keep their decision in every run ({verdict} target {stability_target:.0%})"
        )
        lines += [
            f"    {' -> '.join(d.value for d in u.decisions):<32} {u.unit_id}"
            for u in study.unstable
        ]
    return "\n".join(lines) if lines else "no repeated judgments found"


def bias_json(study: BiasStudy) -> str:
    return json.dumps(dataclasses.asdict(study), indent=2, sort_keys=True) + "\n"


def bias_text(study: BiasStudy) -> str:
    lines: list[str] = []
    if study.length:
        lines.append("Length bias: Spearman rho of grade on word count")
        lines += [
            f"  {b.judge:<14} {b.criterion:<14} {_interval(b.rho)}  n={b.rho.n}  "
            + ("controlling for human grades" if b.controlled else "uncontrolled")
            + ("  EVIDENT" if b.evident else "")
            for b in study.length
        ]
        if not all(b.controlled for b in study.length):
            lines.append(
                "  uncontrolled: too few human ratings, so longer documents that really "
                "are better also raise rho"
            )
    if study.self_preference:
        if lines:
            lines.append("")
        lines.append("Self-preference: own-family grade gap minus a reference judge's (points)")
        lines += [
            f"  {p.judge:<14} vs {p.reference:<14} {p.criterion:<14} {_interval(p.effect)}  "
            f"units={p.effect.n}  gaps {p.judge_gap:+.2f} / {p.reference_gap:+.2f}"
            + ("  EVIDENT" if p.evident else "")
            for p in study.self_preference
        ]
    if study.skipped:
        if lines:
            lines.append("")
        lines.append("Not tested")
        lines += [f"  {reason}" for reason in study.skipped]
    return "\n".join(lines) if lines else "no judged documents found"


def _interval(e: Estimate) -> str:
    return f"{e.value:+6.3f}  [{e.ci_low:+6.3f}, {e.ci_high:+6.3f}]"
