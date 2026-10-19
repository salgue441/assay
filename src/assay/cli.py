"""Command-line entry point. Pipeline stages are added here as they are built."""

from pathlib import Path
from typing import Annotated

import typer

from assay import __version__
from assay.corpus import ManifestError
from assay.orchestrate.pipeline import DEFAULT_CONFIG, build_pipeline
from assay.settings import SettingsError

app = typer.Typer(help="Evaluate LLM-generated documentation and monitor quality drift.")

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Config file.")]


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
    repo: Annotated[str | None, typer.Option(help="Only this repository, as owner/repo.")] = None,
    epoch: Annotated[int | None, typer.Option(min=0, help="Only this epoch.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show Dagster logs.")] = False,
) -> None:
    """Fetch, extract and store the corpus tables for each repository epoch."""

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
        run = pipeline.materialize_corpus(key, log_level="INFO" if verbose else "WARNING")
        failed += not run.succeeded
        counts = ", ".join(f"{rows} {table}" for table, rows in sorted(run.rows.items()))
        typer.echo(f"{'ok' if run.succeeded else 'FAILED':<6} {key:<40} {counts}")

    typer.echo(f"{len(keys) - failed}/{len(keys)} partitions built")
    if failed:
        raise typer.Exit(1)
