"""
Judge bias: whether a judge's grades follow something other than quality.

- Length: does a judge grade longer documents higher? Spearman's rho between
  the judge's grade and the document's word count. When people rated the
  documents, the human grade for the same criterion is partialled out, so
  longer documents that really are better do not count as bias.
- Self-preference: does a judge favour documents written by a model of its
  own family? For each unit documented both by a subject of the judge's
  family and by another subject, the grade gap between the two under the
  judge, minus the same gap under a reference judge of neither family. The
  reference judge absorbs how much better the judge's family really writes.

Bootstraps resample units, since the documents of one unit share its source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from assay.calibrate.bootstrap import Estimate, bootstrap_interval
from assay.generate import Document
from assay.judge import Judgment
from assay.rating import Rating

MIN_PAIRS = 3
MIN_CONTROLLED = 10
"""Human-rated documents needed before length bias is partialled on them."""
CONFIDENCE = 0.95


@dataclass(frozen=True)
class JudgedDocument:
    document_id: str
    unit_id: str
    subject: str
    words: int
    grades: Mapping[str, Mapping[str, float]]
    """Median grade over runs, by judge and then criterion."""
    human: Mapping[str, float]
    """The raters' median grade by criterion; empty when nobody rated it."""


@dataclass(frozen=True)
class LengthBias:
    judge: str
    criterion: str
    rho: Estimate
    """Spearman's rho of grade on word count, partial when `controlled`."""
    controlled: bool
    """Partialled on the human grade for the criterion."""

    @property
    def evident(self) -> bool:
        """The interval excludes zero."""

        return self.rho.ci_low > 0 or self.rho.ci_high < 0


@dataclass(frozen=True)
class SelfPreference:
    judge: str
    family: str
    reference: str
    """The judge of another family whose gap is subtracted."""
    criterion: str
    effect: Estimate
    """Mean over units of the judge's own-family gap minus the reference's,
    in grade points; positive when the judge favours its own family."""
    judge_gap: float
    """Mean grade of own-family documents minus the others', under the judge."""
    reference_gap: float

    @property
    def evident(self) -> bool:
        return self.effect.ci_low > 0 or self.effect.ci_high < 0


@dataclass(frozen=True)
class BiasStudy:
    length: tuple[LengthBias, ...]
    self_preference: tuple[SelfPreference, ...]
    skipped: tuple[str, ...]
    """Checks that could not run, and why."""


def human_grades(
    ratings: Iterable[Rating], *, raters: Sequence[str], rubric_hash: str
) -> dict[str, dict[str, float]]:
    """The median grade of `raters` per document and criterion, under one rubric."""

    grades: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rating in ratings:
        if rating.rater_id in raters and rating.rubric_hash == rubric_hash:
            for score in rating.scores:
                grades[rating.document_id][score.criterion].append(score.score)
    return {
        document: {criterion: median(values) for criterion, values in by_criterion.items()}
        for document, by_criterion in grades.items()
    }


def judged_documents(
    documents: Iterable[Document],
    judgments: Iterable[Judgment],
    human: Mapping[str, Mapping[str, float]] | None = None,
) -> list[JudgedDocument]:
    """The documents with at least one grade in `judgments`, by document id."""

    runs: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for judgment in judgments:
        runs[judgment.document_id][judgment.judge][judgment.criterion].append(judgment.score)
    return sorted(
        (
            JudgedDocument(
                document_id=d.document_id,
                unit_id=d.unit_id,
                subject=d.subject,
                words=len(d.text.split()),
                grades={
                    judge: {c: median(scores) for c, scores in by_criterion.items()}
                    for judge, by_criterion in runs[d.document_id].items()
                },
                human=(human or {}).get(d.document_id, {}),
            )
            for d in documents
            if d.document_id in runs
        ),
        key=lambda d: d.document_id,
    )


def judge_bias(
    documents: Sequence[JudgedDocument],
    *,
    judges: Mapping[str, str],
    subjects: Mapping[str, str],
    criteria: Sequence[str],
    resamples: int,
    seed: int,
) -> BiasStudy:
    """
    Length bias for every judge and criterion, and self-preference for every
    judge sharing a family with a subject, against every judge of neither
    family. `judges` and `subjects` map configured names to model families.
    """

    skipped: list[str] = []
    length = [
        result
        for judge in sorted(judges)
        for criterion in criteria
        if (
            result := _length_bias(
                documents, judge, criterion, resamples=resamples, seed=seed, skipped=skipped
            )
        )
    ]

    preference = []
    for judge, family in sorted(judges.items()):
        own = {s for s, f in subjects.items() if f == family}
        if not own:
            continue
        references = sorted(j for j, f in judges.items() if f != family)
        if not references:
            skipped.append(f"self-preference of {judge}: no judge of another family")
        for reference in references:
            for criterion in criteria:
                outcome = _self_preference(
                    documents,
                    judge=judge,
                    family=family,
                    reference=reference,
                    own=own,
                    criterion=criterion,
                    resamples=resamples,
                    seed=seed,
                )
                if isinstance(outcome, str):
                    skipped.append(outcome)
                else:
                    preference.append(outcome)
    return BiasStudy(tuple(length), tuple(preference), tuple(skipped))


def _length_bias(
    documents: Sequence[JudgedDocument],
    judge: str,
    criterion: str,
    *,
    resamples: int,
    seed: int,
    skipped: list[str],
) -> LengthBias | None:
    graded = [d for d in documents if criterion in d.grades.get(judge, {})]
    rated = [d for d in graded if criterion in d.human]
    controlled = len(rated) >= MIN_CONTROLLED
    sample = rated if controlled else graded
    label = f"length bias of {judge} on {criterion}"
    if len(sample) < MIN_PAIRS:
        skipped.append(f"{label}: {len(sample)} graded documents, need {MIN_PAIRS}")
        return None

    grades = np.array([d.grades[judge][criterion] for d in sample])
    words = np.array([d.words for d in sample], dtype=np.float64)
    control = np.array([d.human[criterion] for d in sample]) if controlled else None

    def statistic(rows: NDArray[np.intp]) -> float:
        return _spearman(grades[rows], words[rows], None if control is None else control[rows])

    value = statistic(np.arange(len(sample), dtype=np.intp))
    if np.isnan(value):
        skipped.append(f"{label}: grades, lengths or human grades are constant")
        return None
    try:
        low, high = bootstrap_interval(
            statistic,
            len(sample),
            groups=[d.unit_id for d in sample],
            resamples=resamples,
            seed=seed,
            confidence=CONFIDENCE,
        )
    except ValueError as exc:
        skipped.append(f"{label}: {exc}")
        return None
    return LengthBias(
        judge=judge,
        criterion=criterion,
        rho=Estimate(value, low, high, n=len(sample), confidence=CONFIDENCE),
        controlled=controlled,
    )


def _spearman(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64] | None
) -> float:
    """Spearman's rho of `x` and `y`, partialled on `z`; NaN when undefined."""

    if np.ptp(x) == 0 or np.ptp(y) == 0 or (z is not None and np.ptp(z) == 0):
        return float("nan")
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rxy = float(np.corrcoef(rx, ry)[0, 1])
    if z is None:
        return rxy
    rz = stats.rankdata(z)
    rxz = float(np.corrcoef(rx, rz)[0, 1])
    ryz = float(np.corrcoef(ry, rz)[0, 1])
    denominator = (1 - rxz**2) * (1 - ryz**2)
    if denominator <= 0:
        return float("nan")
    return (rxy - rxz * ryz) / float(np.sqrt(denominator))


def _self_preference(
    documents: Sequence[JudgedDocument],
    *,
    judge: str,
    family: str,
    reference: str,
    own: set[str],
    criterion: str,
    resamples: int,
    seed: int,
) -> SelfPreference | str:
    """The result, or why it could not be computed."""

    by_unit: dict[str, list[JudgedDocument]] = defaultdict(list)
    for document in documents:
        by_unit[document.unit_id].append(document)

    judge_gaps, reference_gaps = [], []
    for unit_documents in by_unit.values():
        gaps = [_gap(unit_documents, j, own, criterion) for j in (judge, reference)]
        if gaps[0] is not None and gaps[1] is not None:
            judge_gaps.append(gaps[0])
            reference_gaps.append(gaps[1])

    label = f"self-preference of {judge} against {reference} on {criterion}"
    if len(judge_gaps) < MIN_PAIRS:
        return (
            f"{label}: {len(judge_gaps)} units documented by both families and graded by "
            f"both judges, need {MIN_PAIRS}"
        )
    effects = np.array(judge_gaps) - np.array(reference_gaps)
    low, high = bootstrap_interval(
        lambda rows: float(effects[rows].mean()),
        len(effects),
        groups=None,
        resamples=resamples,
        seed=seed,
        confidence=CONFIDENCE,
    )
    return SelfPreference(
        judge=judge,
        family=family,
        reference=reference,
        criterion=criterion,
        effect=Estimate(float(effects.mean()), low, high, n=len(effects), confidence=CONFIDENCE),
        judge_gap=mean(judge_gaps),
        reference_gap=mean(reference_gaps),
    )


def _gap(
    documents: Sequence[JudgedDocument], judge: str, own: set[str], criterion: str
) -> float | None:
    """Mean grade of the own-family documents minus the others', or None."""

    grades: dict[bool, list[float]] = {True: [], False: []}
    for document in documents:
        grade = document.grades.get(judge, {}).get(criterion)
        if grade is not None:
            grades[document.subject in own].append(grade)
    if not grades[True] or not grades[False]:
        return None
    return mean(grades[True]) - mean(grades[False])
