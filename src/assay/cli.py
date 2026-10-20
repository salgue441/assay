"""Command-line entry point. Pipeline stages are added here as they are built."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from assay import __version__
from assay.corpus import ManifestError
from assay.orchestrate.pipeline import DEFAULT_CONFIG, PartitionRun, Pipeline, build_pipeline
from assay.settings import SettingsError

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
        run = materialize(pipeline, key, log_level="INFO" if verbose else "WARNING")
        failed += not run.succeeded
        status = "ok" if run.succeeded else "FAILED"
        details = summarize(run) if run.succeeded else "see --verbose for the error"
        if run.warnings:
            details += f" (warnings: {', '.join(run.warnings)})"
        typer.echo(f"{status:<6} {key:<40} {details}")

    typer.echo(f"{len(keys) - failed}/{len(keys)} partitions done")
    if failed:
        raise typer.Exit(1)
