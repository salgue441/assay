"""Command-line entry point. Pipeline stages are added here as they are built."""

import typer

from assay import __version__

app = typer.Typer(help="Evaluate LLM-generated documentation and monitor quality drift.")


@app.callback()
def main() -> None:
    """Assay command-line interface."""


@app.command()
def version() -> None:
    """Print the installed Assay version."""
    typer.echo(__version__)
