from pathlib import Path

from typer.testing import CliRunner

from assay.cli import app
from assay.corpus import SOURCES
from assay.storage import ParquetStore

from .project import Project


def build_corpus(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["build-corpus", *args])
    return result.exit_code, result.output


def test_builds_every_partition(project: Project) -> None:
    code, output = build_corpus("--config", str(project.config))

    assert code == 0, output
    assert "ok     owner__greeter@00" in output
    assert "ok     owner__greeter@01" in output
    assert "2/2 partitions built" in output
    assert len(ParquetStore(project.data).read(SOURCES)) == 2


def test_filters_by_repository_and_epoch(project: Project) -> None:
    code, output = build_corpus(
        "--config", str(project.config), "--repo", "owner/greeter", "--epoch", "1"
    )

    assert code == 0, output
    assert "1 sources, 3 symbols, 3 units" in output
    assert "1/1 partitions built" in output


def test_reports_when_nothing_matches(project: Project) -> None:
    code, output = build_corpus("--config", str(project.config), "--repo", "owner/missing")

    assert code == 2
    assert "no repository epoch matches" in output


def test_reports_an_invalid_config(tmp_path: Path) -> None:
    code, output = build_corpus("--config", str(tmp_path / "missing.yaml"))

    assert code == 2
    assert "missing.yaml" in output
