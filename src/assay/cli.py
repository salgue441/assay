"""Command-line entry point. Pipeline stages are added here as they are built."""

import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

import assay.rating
from assay import __version__
from assay.calibrate import RatedDocument, Routing, calibrate, rated_documents
from assay.calibrate.golden import freeze_golden, read_active_golden, select_golden, write_golden
from assay.calibrate.report import report_json, report_text, variance_json, variance_text
from assay.calibrate.reveals import Reveal, conflicting, read_reveals, record_reveal
from assay.calibrate.study import split_documents
from assay.calibrate.variance import judge_variance
from assay.corpus import SOURCES, ManifestError
from assay.generate import DOCUMENTS, Document
from assay.judge import JUDGMENTS
from assay.orchestrate.assets import scoring_judgments
from assay.orchestrate.pipeline import DEFAULT_CONFIG, PartitionRun, Pipeline, build_pipeline
from assay.rating import RATER_ID_PATTERN, RATINGS
from assay.scoring import SCORES, ScorerConfig
from assay.settings import EvaluationSettings, Settings, SettingsError, load_settings
from assay.storage import ParquetStore

app = typer.Typer(help="Evaluate LLM-generated documentation and monitor quality drift.")

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Config file.")]
RepoOption = Annotated[str | None, typer.Option(help="Only this repository, as owner/repo.")]
EpochOption = Annotated[int | None, typer.Option(min=0, help="Only this epoch.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show Dagster logs.")]


@app.callback()
def main() -> None:
    """Assay command-line interface."""


@app.command()
def version() -> None:
    """Print the installed Assay version."""
    typer.echo(__version__)


@app.command("build-corpus")
def build_corpus(
    config: ConfigOption = DEFAULT_CONFIG,
    repo: RepoOption = None,
    epoch: EpochOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Fetch, extract and store the corpus tables for each repository epoch."""

    _run_partitions(
        config,
        repo=repo,
        epoch=epoch,
        verbose=verbose,
        materialize=Pipeline.materialize_corpus,
        summarize=lambda run: ", ".join(
            f"{counts['rows']} {table}" for table, counts in sorted(run.metadata.items())
        ),
    )


@app.command()
def generate(
    config: ConfigOption = DEFAULT_CONFIG,
    repo: RepoOption = None,
    epoch: EpochOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Document each repository epoch's sampled units with every subject model."""

    def summarize(run: PartitionRun) -> str:
        counts = run.metadata.get("documents", {})
        return (
            f"{counts.get('new', 0)} new, {counts.get('skipped', 0)} existing, "
            f"{counts.get('failed', 0)} failed, {counts.get('rows', 0)} stored; "
            f"{counts.get('too_large', 0)} units too large to prompt"
        )

    _run_partitions(
        config,
        repo=repo,
        epoch=epoch,
        verbose=verbose,
        materialize=Pipeline.materialize_documents,
        summarize=summarize,
    )


@app.command()
def measure(
    config: ConfigOption = DEFAULT_CONFIG,
    repo: RepoOption = None,
    epoch: EpochOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Compute the static metrics for each repository epoch's documents."""

    def summarize(run: PartitionRun) -> str:
        counts = run.metadata.get("metrics", {})
        return (
            f"{counts.get('rows', 0)} metric values for {counts.get('documents', 0)} documents, "
            f"{counts.get('critical', 0)} critical"
        )

    _run_partitions(
        config,
        repo=repo,
        epoch=epoch,
        verbose=verbose,
        materialize=Pipeline.materialize_metrics,
        summarize=summarize,
    )


@app.command()
def judge(
    config: ConfigOption = DEFAULT_CONFIG,
    repo: RepoOption = None,
    epoch: EpochOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Grade each repository epoch's documents with every judge model."""

    def summarize(run: PartitionRun) -> str:
        counts = run.metadata.get("judgments", {})
        return (
            f"{counts.get('new', 0)} new, {counts.get('skipped', 0)} existing, "
            f"{counts.get('failed', 0)} failed, {counts.get('rows', 0)} stored; "
            f"{counts.get('too_large', 0)} documents too large, "
            f"{counts.get('contradictions', 0)} top scores despite issues"
        )

    _run_partitions(
        config,
        repo=repo,
        epoch=epoch,
        verbose=verbose,
        materialize=Pipeline.materialize_judgments,
        summarize=summarize,
    )


@app.command()
def score(
    config: ConfigOption = DEFAULT_CONFIG,
    repo: RepoOption = None,
    epoch: EpochOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Score and route each repository epoch's documents from their metrics and judgments."""

    def summarize(run: PartitionRun) -> str:
        counts = run.metadata.get("scores", {})
        decisions = ", ".join(
            f"{decision} {counts.get(decision, 0)}"
            for decision in ("publish", "revise", "escalate", "reject")
        )
        return (
            f"{counts.get('scored', 0)} scored ({decisions}), {counts.get('unscored', 0)} unscored"
        )

    _run_partitions(
        config,
        repo=repo,
        epoch=epoch,
        verbose=verbose,
        materialize=Pipeline.materialize_scores,
        summarize=summarize,
    )


@app.command()
def rate(
    rater: Annotated[str, typer.Option(help="Your pseudonym, such as r1. Never your name.")],
    config: ConfigOption = DEFAULT_CONFIG,
    additional: Annotated[
        bool,
        typer.Option(
            "--additional", help="Rate only the shared overlap subset, as a rater after the first."
        ),
    ] = False,
) -> None:
    """Open the blind rating app in your browser."""

    if not re.fullmatch(RATER_ID_PATTERN, rater):
        typer.echo(
            f"error: rater must be a short lowercase pseudonym such as r1, not {rater!r}", err=True
        )
        raise typer.Exit(2)
    if importlib.util.find_spec("streamlit") is None:
        typer.echo("error: the rating app needs the app extra: uv sync --extra app", err=True)
        raise typer.Exit(2)
    try:
        settings = load_settings(config)
    except SettingsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    share = settings.rating.additional_rater_share if additional else 1.0
    app_path = Path(assay.rating.__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        # Only this machine can reach the app, and Streamlit sends no usage data.
        "--server.address",
        "localhost",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--config",
        str(config),
        "--rater",
        rater,
        "--share",
        str(share),
    ]
    raise typer.Exit(subprocess.run(command, check=False).returncode)


@app.command("calibrate")
def calibrate_command(
    config: ConfigOption = DEFAULT_CONFIG,
    *,
    rater: Annotated[
        list[str] | None,
        typer.Option(
            help="Calibrate against this rater instead of evaluation.raters; repeatable. "
            "Use it to inspect other raters, never to report human agreement."
        ),
    ] = None,
    test: Annotated[
        bool,
        typer.Option("--test", help="Also evaluate the held-out test split. Do this once."),
    ] = False,
    repeat_test: Annotated[
        bool,
        typer.Option(
            "--repeat-test",
            help="Evaluate the test split although it was evaluated under another "
            "configuration. Recorded in reports/test_reveals.jsonl.",
        ),
    ] = False,
    resamples: Annotated[
        int | None,
        typer.Option(min=1, help="Bootstrap resamples, instead of evaluation.bootstrap_resamples."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Where to write the JSON report. [default: <storage>/reports/]"),
    ] = None,
) -> None:
    """Compare the composite score with human ratings and report the agreement."""

    loaded = _load_rated(config, rater)
    settings, evaluation, scorer = loaded.settings, loaded.evaluation, loaded.scorer
    documents = loaded.documents

    rated = {d.document_id for d in documents}
    texts = {d.document_id: d.text for d in loaded.stored if d.document_id in rated}
    try:
        report = calibrate(
            documents,
            prior=settings.scoring.weights,
            routing=_routing(settings),
            development_fraction=evaluation.split.development_fraction,
            shrink_to_prior=evaluation.weight_fit.shrink_to_prior,
            resamples=resamples or evaluation.bootstrap_resamples,
            seed=settings.run.seed,
            include_test=test,
            texts=texts,
            stratify_by=evaluation.split.stratify_by,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if test:
        ledger = settings.storage.root / "reports" / "test_reveals.jsonl"
        reveal = Reveal(
            revealed_at=datetime.now(UTC),
            scorer_version=scorer.version,
            rubric_hash=loaded.rubric_hash,
            raters=report.raters,
            test_repos=report.test_repos,
        )
        earlier = conflicting(read_reveals(ledger), reveal)
        if earlier and not repeat_test:
            typer.echo(
                "error: the test split was already evaluated under another configuration:\n"
                + "\n".join(f"  {r.describe()}" for r in earlier)
                + "\nEvaluating it again after changes it informed leaks the test set. "
                "Pass --repeat-test to do it anyway; it is recorded.",
                err=True,
            )
            raise typer.Exit(1)
        record_reveal(ledger, reveal)

    path = output or settings.storage.root / "reports" / f"calibration-{scorer.version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json(report), encoding="utf-8")
    typer.echo(
        report_text(
            report,
            rho_target=evaluation.targets.spearman_rho,
            kappa_target=evaluation.targets.weighted_kappa,
        )
    )
    typer.echo(f"\nreport written to {path}")


@app.command("golden-set")
def golden_set_command(
    config: ConfigOption = DEFAULT_CONFIG,
    replace: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="Freeze a new set although another is active. Monitoring then restarts "
            "from the new set.",
        ),
    ] = False,
) -> None:
    """Freeze the golden set: rated development documents that drift monitoring re-scores."""

    loaded = _load_rated(config, None)
    settings, evaluation = loaded.settings, loaded.evaluation
    try:
        split = split_documents(
            loaded.documents,
            development_fraction=evaluation.split.development_fraction,
            seed=settings.run.seed,
            stratify_by=evaluation.split.stratify_by,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    size = evaluation.golden_set.size
    chosen = select_golden(
        split.dev, size=size, seed=settings.run.seed, stratify_by=evaluation.split.stratify_by
    )
    golden = freeze_golden(
        chosen,
        frozen_at=datetime.now(UTC),
        scorer_version=loaded.scorer.version,
        rubric_hash=loaded.rubric_hash,
    )
    directory = settings.storage.root / "golden_set"
    active = read_active_golden(directory)
    if active is not None and active.version == golden.version:
        typer.echo(f"golden set {golden.version} is already active; nothing changed")
        return
    if active is not None and not replace:
        typer.echo(
            f"error: golden set {active.version} ({len(active.documents)} documents, frozen "
            f"{active.frozen_at:%Y-%m-%d}) is active. Pass --replace to freeze a new one.",
            err=True,
        )
        raise typer.Exit(1)

    path = write_golden(directory, golden)
    short = f" (only {len(chosen)} rated development documents, of {size} wanted)"
    typer.echo(
        f"froze golden set {golden.version}: {len(chosen)} documents from "
        f"{', '.join(sorted({d.repo for d in chosen}))}" + (short if len(chosen) < size else "")
    )
    typer.echo(f"written to {path}")


@app.command("judge-variance")
def judge_variance_command(
    config: ConfigOption = DEFAULT_CONFIG,
    output: Annotated[
        Path | None,
        typer.Option(help="Where to write the JSON report. [default: <storage>/reports/]"),
    ] = None,
) -> None:
    """Measure how much the scoring judge's grades and routing move between runs."""

    try:
        settings = load_settings(config)
    except SettingsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    scorer = _scorer_config(settings)
    store = ParquetStore(settings.storage.root)
    scores = [s for s in store.read(SCORES) if s.scorer_version == scorer.version]
    studies = judge_variance(
        scores,
        scoring_judgments(store.read(JUDGMENTS), settings, any_temperature=True),
        weights=settings.scoring.weights,
        criteria=settings.judge.criteria,
        routing=_routing(settings),
    )
    path = output or settings.storage.root / "reports" / f"judge-variance-{scorer.version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(variance_json(studies), encoding="utf-8")
    target = settings.evaluation.targets.judge_band_stability if settings.evaluation else 0.9
    typer.echo(variance_text(studies, stability_target=target))
    typer.echo(f"\nreport written to {path}")


def _scorer_config(settings: Settings) -> ScorerConfig:
    scoring = settings.scoring
    return ScorerConfig(
        weights=scoring.weights,
        judge=scoring.judge,
        publish_at_or_above=scoring.routing.publish_at_or_above,
        escalate_below=scoring.routing.escalate_below,
    )


def _routing(settings: Settings) -> Routing:
    routing = settings.scoring.routing
    return Routing(
        publish_at_or_above=routing.publish_at_or_above, escalate_below=routing.escalate_below
    )


@dataclass(frozen=True)
class _Rated:
    """What calibration needs: settings, the current scorer and the rated documents."""

    settings: Settings
    evaluation: EvaluationSettings
    scorer: ScorerConfig
    rubric_hash: str
    stored: list[Document]
    documents: list[RatedDocument]


def _load_rated(config: Path, rater: list[str] | None) -> _Rated:
    """
    The documents scored by the current scorer and rated by `rater`, or by
    the configured human raters. Exits 2 for an invalid config or rater and
    1 when nothing is rated.
    """

    try:
        settings = load_settings(config)
        rubric_hash = settings.judge.rubric().hash
    except SettingsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    evaluation = settings.evaluation
    if evaluation is None:
        typer.echo("error: calibrating needs an evaluation section in the config", err=True)
        raise typer.Exit(2)
    raters = tuple(rater) if rater else evaluation.raters
    invalid = [r for r in raters if not re.fullmatch(RATER_ID_PATTERN, r)]
    if invalid:
        typer.echo(f"error: raters must be short lowercase pseudonyms: {invalid}", err=True)
        raise typer.Exit(2)
    unlisted = sorted(set(raters) - set(evaluation.raters))
    if unlisted:
        typer.echo(
            f"warning: {', '.join(unlisted)} not in evaluation.raters; "
            "these results are not human agreement",
            err=True,
        )

    scorer = _scorer_config(settings)
    store = ParquetStore(settings.storage.root)
    stored = store.read(DOCUMENTS)
    languages = {(s.repo, s.commit): s.language.value for s in store.read(SOURCES)}
    documents = rated_documents(
        store.read(SCORES),
        store.read(RATINGS),
        raters=raters,
        scorer_version=scorer.version,
        rubric_hash=rubric_hash,
        strata={
            d.document_id: {
                "language": languages.get((d.repo, d.commit), "unknown"),
                "unit_kind": d.kind.value,
            }
            for d in stored
        },
    )
    if not documents:
        typer.echo(
            f"error: no document scored by scorer {scorer.version} is rated by "
            f"{', '.join(raters)} under rubric {settings.judge.rubric_version}",
            err=True,
        )
        raise typer.Exit(1)

    return _Rated(settings, evaluation, scorer, rubric_hash, stored, documents)


def _run_partitions(
    config: Path,
    *,
    repo: str | None,
    epoch: int | None,
    verbose: bool,
    materialize: Callable[..., PartitionRun],
    summarize: Callable[[PartitionRun], str],
) -> None:
    """
    Materialize every selected partition and print one line each.

    Exits 1 if any partition fails and 2 for an invalid config or an empty
    selection. Failed non-blocking checks are shown but do not fail the run.
    A failure prints its reason on the partition's line; Dagster's own logs
    appear only with --verbose.
    """

    try:
        pipeline = build_pipeline(config)
    except (SettingsError, ManifestError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    keys = pipeline.partitions.keys(repo=repo, epoch=epoch)
    if not keys:
        typer.echo("error: no repository epoch matches the filters", err=True)
        raise typer.Exit(2)

    failed = 0
    for key in keys:
        run = materialize(pipeline, key, log_level="INFO" if verbose else "CRITICAL")
        failed += not run.succeeded
        status = "ok" if run.succeeded else "FAILED"
        details = summarize(run) if run.succeeded else (run.error or "unknown error")
        if run.warnings:
            details += f" (warnings: {', '.join(run.warnings)})"
        typer.echo(f"{status:<6} {key:<40} {details}")

    typer.echo(f"{len(keys) - failed}/{len(keys)} partitions done")
    if failed:
        raise typer.Exit(1)
