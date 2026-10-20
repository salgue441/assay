"""Command-line entry point. Pipeline stages are added here as they are built."""

import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

import assay.rating
from assay import __version__
from assay.corpus import ManifestError
from assay.orchestrate.pipeline import DEFAULT_CONFIG, PartitionRun, Pipeline, build_pipeline
from assay.rating import RATER_ID_PATTERN
from assay.settings import SettingsError, load_settings

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
