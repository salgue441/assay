"""
The calibration study: join each document's composite score with the human
ratings of it, then measure how well the score orders documents the way the
raters do. Pure: callers pass the rows read from storage.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, median

import numpy as np
from scipy import stats

from assay.calibrate.ablation import AblationStudy, ablate, composite
from assay.calibrate.agreement import krippendorff_alpha, weighted_kappa_estimate
from assay.calibrate.baseline import MODELS, feature_matrix, fit_and_predict, out_of_fold
from assay.calibrate.bootstrap import Estimate
from assay.calibrate.correlation import RankMethod, rank_correlation
from assay.calibrate.decisions import DecisionQuality, Routing, choose_routing, routed
from assay.calibrate.split import Split, grouped_split, stratified_grouped_split
from assay.calibrate.weights import choose_shrinkage, fit_weights
from assay.judge import MAX_SCORE, MIN_SCORE
from assay.rating import Decision, Rating
from assay.scoring import RoutingDecision, Score

SHRINKAGE_CANDIDATES = (0.0, 0.01, 0.1, 1.0, 10.0)
"""Candidate strengths of the pull toward the prior weights."""

OVERALL = "overall"
CONFIGURED = "configured"
CHOSEN = "chosen"
MAX_FOLDS = 5
DISAGREEMENTS_SHOWN = 10
METHODS: tuple[RankMethod, ...] = ("spearman", "kendall")


@dataclass(frozen=True)
class RatedDocument:
    """One scored document with the consensus of the raters who rated it."""

    document_id: str
    unit_id: str
    subject: str
    repo: str
    composite: float
    decision: RoutingDecision
    components: Mapping[str, float]
    ratings: Mapping[str, Mapping[str, int]]
    """Each rater's ratings by rater id: "overall" and every criterion."""
    overall: float
    """Mean overall rating across raters, on the 1-5 scale."""
    criteria: Mapping[str, int]
    """Median rating per criterion across raters, rounded half up."""
    human_decision: Decision | None
    """The raters' shared decision, or None when they disagree."""
    strata: Mapping[str, str] = field(default_factory=dict)
    """Values of the variables the split balances, such as language."""

    @property
    def rejected(self) -> bool:
        return self.decision is RoutingDecision.REJECT

    def judge_grade(self, criterion: str) -> int | None:
        """The judge's grade on the rubric scale, if the criterion was judged."""

        value = self.components.get(criterion)
        if value is None:
            return None
        return _round_half_up(MIN_SCORE + value * (MAX_SCORE - MIN_SCORE))


def rated_documents(
    scores: Iterable[Score],
    ratings: Iterable[Rating],
    *,
    raters: Sequence[str],
    scorer_version: str,
    rubric_hash: str,
    strata: Mapping[str, Mapping[str, str]] | None = None,
) -> list[RatedDocument]:
    """
    The documents scored under `scorer_version` and rated by at least one
    of `raters` under the rubric with `rubric_hash`, sorted by document id.
    Ratings by anyone not in `raters` are ignored. `strata` gives each
    document's stratum values by document id.
    """

    allowed = set(raters)
    by_document: dict[str, list[Rating]] = defaultdict(list)
    for rating in ratings:
        if rating.rater_id in allowed and rating.rubric_hash == rubric_hash:
            by_document[rating.document_id].append(rating)

    documents = []
    for score in scores:
        group = by_document.get(score.document_id)
        if score.scorer_version != scorer_version or not group:
            continue
        criteria = sorted({s.criterion for r in group for s in r.scores})
        decisions = {r.decision for r in group}
        documents.append(
            RatedDocument(
                document_id=score.document_id,
                unit_id=score.unit_id,
                subject=score.subject,
                repo=score.repo,
                composite=score.composite,
                decision=score.decision,
                components={c.name: c.value for c in score.components},
                ratings={
                    r.rater_id: {OVERALL: r.overall, **{s.criterion: s.score for s in r.scores}}
                    for r in group
                },
                overall=mean(r.overall for r in group),
                criteria={
                    name: _round_half_up(median(r.score_for(name) for r in group))
                    for name in criteria
                },
                human_decision=decisions.pop() if len(decisions) == 1 else None,
                strata=(strata or {}).get(score.document_id, {}),
            )
        )
    return sorted(documents, key=lambda d: d.document_id)


@dataclass(frozen=True)
class Disagreement:
    """A document the score and the raters place far apart."""

    document_id: str
    unit_id: str
    subject: str
    repo: str
    composite: float
    overall: float
    rank_gap: float
    """Score percentile minus human percentile, in [-1, 1]; positive when the
    score ranks the document higher than the raters do."""


@dataclass(frozen=True)
class FittedWeights:
    weights: Mapping[str, float]
    shrinkage: float
    documents: int
    """Development documents with every weighted component, used to fit."""


@dataclass(frozen=True)
class CalibrationReport:
    """
    The study's results. Test-split results are present only when the
    caller asked for them, since the test split is meant to be used once.
    """

    raters: tuple[str, ...]
    documents: int
    dev_repos: tuple[str, ...]
    test_repos: tuple[str, ...]
    dev_documents: int
    test_documents: int
    rater_alpha: Mapping[str, float]
    """Krippendorff's alpha (ordinal) across raters, overall and per criterion."""
    correlation: Mapping[str, Mapping[str, Estimate]]
    """Split -> method -> composite (prior weights) against human overall,
    over non-rejected documents."""
    fitted: FittedWeights | None
    fitted_correlation: Mapping[str, Mapping[str, Estimate]]
    """Split -> method -> composite under the fitted weights."""
    judge_kappa: Mapping[str, Mapping[str, Estimate]]
    """Split -> criterion -> quadratic-weighted kappa, judge against humans."""
    decisions: Mapping[str, Mapping[str, Mapping[str, int]]]
    """Split -> scorer decision -> human decision ("disputed" if raters
    disagree) -> count."""
    chosen_routing: Routing | None
    """Thresholds that best match the raters' decisions on development;
    None when no development document has an undisputed human decision."""
    chosen_decisions: Mapping[str, Mapping[str, Mapping[str, int]]]
    """Like `decisions`, routed by `chosen_routing`."""
    decision_quality: Mapping[str, Mapping[str, DecisionQuality]]
    """Split -> "configured" or "chosen" thresholds -> agreement with the
    raters' decisions and publish precision."""
    ablations: Mapping[str, AblationStudy]
    """Split -> the scorer under the prior weights with each part removed."""
    baseline: Mapping[str, Mapping[str, Estimate]]
    """Split -> predictor -> Spearman rho against human overall over
    non-rejected documents: the composite, the judge alone and each learned
    baseline model. Development values of the models are out-of-fold."""
    disagreements: tuple[Disagreement, ...]
    """The largest score-human rank gaps on the development split."""
    notes: tuple[str, ...] = field(default=())
    """Results that could not be computed, and why."""


def calibrate(
    documents: Sequence[RatedDocument],
    *,
    prior: Mapping[str, float],
    routing: Routing,
    development_fraction: float,
    shrink_to_prior: bool,
    resamples: int,
    seed: int,
    include_test: bool,
    texts: Mapping[str, str],
    stratify_by: Sequence[str] = (),
) -> CalibrationReport:
    """
    Split `documents` by repository, fit weights on the development side
    and measure agreement with the raters on each reported side. `texts`
    holds each document's text by id, for the learned baseline; `stratify_by`
    names the stratum variables the split balances. A result
    that cannot be computed, such as a correlation over constant ratings,
    is left out and explained in `notes`. Raises `ValueError` when the
    documents come from fewer than two repositories or lack a stratum value.
    """

    repos = {d.repo for d in documents}
    if len(repos) < 2:
        msg = (
            f"calibration needs rated documents from at least two repositories, one for "
            f"each split; got {len(documents)} documents from {len(repos)}"
        )
        raise ValueError(msg)

    notes: list[str] = []
    split = split_documents(
        documents, development_fraction=development_fraction, seed=seed, stratify_by=stratify_by
    )
    sides = {"dev": split.dev, **({"test": split.test} if include_test else {})}

    fitted = _fit(split.dev, prior, shrink_to_prior=shrink_to_prior, seed=seed, notes=notes)
    chosen = choose_routing(split.dev)
    if chosen is None:
        notes.append("thresholds: no development document has an undisputed human decision")
    criteria = sorted({c for d in documents for c in d.criteria})

    correlation: dict[str, dict[str, Estimate]] = {}
    fitted_correlation: dict[str, dict[str, Estimate]] = {}
    judge_kappa: dict[str, dict[str, Estimate]] = {}
    for side, docs in sides.items():
        kept = [d for d in docs if not d.rejected]
        if len({d.repo for d in docs}) == 1:
            notes.append(
                f"{side} intervals: one repository, so the bootstrap resamples documents "
                "and ignores how documents of a repository resemble each other"
            )
        refit: list[tuple[RatedDocument, float]] = []
        if fitted is not None:
            fitted_correlation[side] = {}
            refit = [(d, c) for d in kept if (c := composite(d, fitted.weights)) is not None]
        correlation[side] = {}
        for method in METHODS:
            estimate = _attempt(
                notes,
                f"{side} {method}",
                _correlate,
                [d.composite for d in kept],
                kept,
                method,
                resamples,
                seed,
            )
            if estimate is not None:
                correlation[side][method] = estimate
            if fitted is not None:
                estimate = _attempt(
                    notes,
                    f"{side} fitted {method}",
                    _correlate,
                    [c for _, c in refit],
                    [d for d, _ in refit],
                    method,
                    resamples,
                    seed,
                )
                if estimate is not None:
                    fitted_correlation[side][method] = estimate
        judge_kappa[side] = {}
        for criterion in criteria:
            estimate = _attempt(
                notes, f"{side} kappa {criterion}", _kappa, docs, criterion, resamples, seed
            )
            if estimate is not None:
                judge_kappa[side][criterion] = estimate

    baseline = _baseline(
        split,
        sides,
        texts=texts,
        prior=prior,
        criteria=criteria,
        resamples=resamples,
        seed=seed,
        notes=notes,
    )
    raters = sorted({r for d in documents for r in d.ratings})
    return CalibrationReport(
        raters=tuple(raters),
        documents=len(documents),
        dev_repos=tuple(sorted({d.repo for d in split.dev})),
        test_repos=tuple(sorted({d.repo for d in split.test})),
        dev_documents=len(split.dev),
        test_documents=len(split.test),
        rater_alpha=_rater_alpha(documents, [OVERALL, *criteria], notes) if len(raters) > 1 else {},
        correlation=correlation,
        fitted=fitted,
        fitted_correlation=fitted_correlation,
        judge_kappa=judge_kappa,
        decisions={side: _confusion(docs) for side, docs in sides.items()},
        chosen_routing=chosen,
        chosen_decisions=(
            {side: _confusion(docs, chosen) for side, docs in sides.items()} if chosen else {}
        ),
        decision_quality={
            side: {
                CONFIGURED: routed(docs, routing),
                **({CHOSEN: routed(docs, chosen)} if chosen else {}),
            }
            for side, docs in sides.items()
        },
        ablations={
            side: ablate(docs, weights=prior, judged=criteria, routing=routing)
            for side, docs in sides.items()
        },
        baseline=baseline,
        disagreements=_disagreements(split.dev),
        notes=tuple(notes),
    )


def split_documents(
    documents: Sequence[RatedDocument],
    *,
    development_fraction: float,
    seed: int,
    stratify_by: Sequence[str] = (),
) -> Split[RatedDocument]:
    """
    The development/test split the study uses: by repository, balancing
    `stratify_by` when given. Raises `ValueError` when a document lacks a
    stratum value.
    """

    unstratified = [
        d.document_id[:12] for d in documents if any(v not in d.strata for v in stratify_by)
    ]
    if unstratified:
        msg = f"documents without {', '.join(stratify_by)}: {', '.join(unstratified[:5])}"
        raise ValueError(msg)
    if stratify_by:
        return stratified_grouped_split(
            documents,
            group_of=lambda d: d.repo,
            strata_of=lambda d: {v: d.strata[v] for v in stratify_by},
            test_fraction=1 - development_fraction,
            seed=seed,
        )
    return grouped_split(
        documents, group_of=lambda d: d.repo, test_fraction=1 - development_fraction, seed=seed
    )


def _correlate(
    scores: Sequence[float],
    documents: Sequence[RatedDocument],
    method: RankMethod,
    resamples: int,
    seed: int,
) -> Estimate:
    return rank_correlation(
        scores,
        [d.overall for d in documents],
        method=method,
        groups=_clusters([d.repo for d in documents]),
        resamples=resamples,
        seed=seed,
    )


def _clusters(repos: Sequence[str]) -> Sequence[str] | None:
    """Bootstrap clusters: whole repositories, or single documents when
    there is only one repository to draw."""

    return repos if len(set(repos)) > 1 else None


def _kappa(
    documents: Sequence[RatedDocument], criterion: str, resamples: int, seed: int
) -> Estimate:
    pairs = [
        (grade, d.criteria[criterion], d.repo)
        for d in documents
        if criterion in d.criteria and (grade := d.judge_grade(criterion)) is not None
    ]
    if not pairs:
        msg = "no document has both a judge grade and a human rating"
        raise ValueError(msg)
    judge, human, repos = zip(*pairs, strict=True)
    return weighted_kappa_estimate(
        judge, human, groups=_clusters(repos), resamples=resamples, seed=seed
    )


def _fit(
    documents: Sequence[RatedDocument],
    prior: Mapping[str, float],
    *,
    shrink_to_prior: bool,
    seed: int,
    notes: list[str],
) -> FittedWeights | None:
    names = sorted(name for name, weight in prior.items() if weight > 0)
    complete = [d for d in documents if all(name in d.components for name in names)]
    repos = sorted({d.repo for d in complete})
    if len(complete) < len(names) + 1:
        notes.append(
            f"weight fit: {len(complete)} development documents have every weighted "
            f"component; at least {len(names) + 1} are needed"
        )
        return None

    features = np.array([[d.components[name] for name in names] for d in complete])
    target = np.array([(d.overall - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) for d in complete])
    prior_weights = np.array([prior[name] for name in names])
    prior_weights = prior_weights / prior_weights.sum()

    shrinkage = 0.0
    if shrink_to_prior:
        if len(repos) < 2:
            notes.append(
                "weight fit: choosing the shrinkage needs at least two development "
                f"repositories, got {len(repos)}; prior weights are kept"
            )
            return None
        shrinkage = choose_shrinkage(
            features,
            target,
            groups=[d.repo for d in complete],
            prior=prior_weights,
            candidates=SHRINKAGE_CANDIDATES,
            folds=min(MAX_FOLDS, len(repos)),
            seed=seed,
        )
    weights = fit_weights(features, target, prior=prior_weights, shrinkage=shrinkage)
    return FittedWeights(
        weights=dict(zip(names, (float(w) for w in weights), strict=True)),
        shrinkage=shrinkage,
        documents=len(complete),
    )


def _rater_alpha(
    documents: Sequence[RatedDocument], measures: Sequence[str], notes: list[str]
) -> dict[str, float]:
    """Krippendorff's alpha (ordinal) per measure, over every document."""

    alpha = {}
    for measure in measures:
        ratings = {
            d.document_id: {rater: values.get(measure) for rater, values in d.ratings.items()}
            for d in documents
        }
        try:
            alpha[measure] = krippendorff_alpha(ratings, level="ordinal")
        except ValueError as exc:
            notes.append(f"rater alpha {measure}: {exc}")
    return alpha


def _confusion(
    documents: Sequence[RatedDocument], routing: Routing | None = None
) -> dict[str, dict[str, int]]:
    """Scorer decision -> human decision -> count, under the stored decisions
    or, given `routing`, under those thresholds."""

    counts: Counter[tuple[str, str]] = Counter(
        (
            (routing.route(d.composite, critical=d.rejected) if routing else d.decision).value,
            d.human_decision.value if d.human_decision else "disputed",
        )
        for d in documents
    )
    table: dict[str, dict[str, int]] = defaultdict(dict)
    for (scorer, human), count in sorted(counts.items()):
        table[scorer][human] = count
    return dict(table)


def _disagreements(documents: Sequence[RatedDocument]) -> tuple[Disagreement, ...]:
    if len(documents) < 2:
        return ()
    score_pct = _percentiles([d.composite for d in documents])
    human_pct = _percentiles([d.overall for d in documents])
    gaps = [
        Disagreement(
            document_id=d.document_id,
            unit_id=d.unit_id,
            subject=d.subject,
            repo=d.repo,
            composite=d.composite,
            overall=d.overall,
            rank_gap=float(s - h),
        )
        for d, s, h in zip(documents, score_pct, human_pct, strict=True)
    ]
    gaps.sort(key=lambda g: (-abs(g.rank_gap), g.document_id))
    return tuple(gaps[:DISAGREEMENTS_SHOWN])


def _percentiles(values: Sequence[float]) -> list[float]:
    """Average ranks scaled to [0, 1]."""

    ranks = stats.rankdata(values)
    return [float((r - 1) / (len(values) - 1)) for r in ranks]


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _attempt[**P, T](
    notes: list[str], label: str, compute: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T | None:
    """`compute(*args, **kwargs)`, or None with the reason added to `notes`
    when it raises `ValueError`."""

    try:
        return compute(*args, **kwargs)
    except ValueError as exc:
        notes.append(f"{label}: {exc}")
        return None


def _baseline(
    split: Split[RatedDocument],
    sides: Mapping[str, Sequence[RatedDocument]],
    *,
    texts: Mapping[str, str],
    prior: Mapping[str, float],
    criteria: Sequence[str],
    resamples: int,
    seed: int,
    notes: list[str],
) -> dict[str, dict[str, Estimate]]:
    """
    Rank correlation with human overall of the composite, the judge alone
    and each learned baseline, over the non-rejected documents of each side.
    The models train on development; their development predictions are
    out-of-fold.
    """

    judge_weights = {name: prior[name] for name in criteria if name in prior}
    predicted = _model_predictions(
        split,
        sides,
        texts=texts,
        metrics=sorted(name for name in prior if name not in criteria),
        seed=seed,
        notes=notes,
    )

    results: dict[str, dict[str, Estimate]] = {}
    for side, docs in sides.items():
        candidates: dict[str, list[float | None]] = {
            "composite": [d.composite for d in docs],
            "judge_alone": [composite(d, judge_weights) for d in docs],
            **predicted[side],
        }
        results[side] = {}
        for name, scores in candidates.items():
            pairs = [(s, d) for s, d in zip(scores, docs, strict=True) if s is not None]
            kept = [(s, d) for s, d in pairs if not d.rejected]
            estimate = _attempt(
                notes,
                f"{side} baseline {name}",
                _correlate,
                [s for s, _ in kept],
                [d for _, d in kept],
                "spearman",
                resamples,
                seed,
            )
            if estimate is not None:
                results[side][name] = estimate
    return results


def _model_predictions(
    split: Split[RatedDocument],
    sides: Mapping[str, Sequence[RatedDocument]],
    *,
    texts: Mapping[str, str],
    metrics: Sequence[str],
    seed: int,
    notes: list[str],
) -> dict[str, dict[str, list[float | None]]]:
    """Each baseline model's predictions per side: out-of-fold on
    development, and fitted on development for test."""

    predicted: dict[str, dict[str, list[float | None]]] = {side: {} for side in sides}
    untexted = sum(d.document_id not in texts for docs in sides.values() for d in docs)
    if untexted:
        notes.append(f"baseline models: {untexted} documents have no stored text")
        return predicted

    features, _ = feature_matrix(split.dev, texts, metrics)
    target = np.array([d.overall for d in split.dev], dtype=np.float64)
    groups = [d.repo for d in split.dev]
    test_features = feature_matrix(sides["test"], texts, metrics)[0] if "test" in sides else None
    for model in MODELS:
        oof = _attempt(
            notes, f"dev {model}", out_of_fold, features, target, groups, model=model, seed=seed
        )
        if oof is not None:
            predicted["dev"][model] = [float(p) for p in oof]
        if test_features is not None:
            test = fit_and_predict(features, target, groups, test_features, model=model, seed=seed)
            predicted["test"][model] = [float(p) for p in test]
    return predicted
