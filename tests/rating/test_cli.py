import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assay.cli import app

from .fixtures import write_config


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the commands `assay rate` would start instead of starting them."""

    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return commands


def test_starts_the_app_for_the_primary_rater(tmp_path: Path, launched: list[list[str]]) -> None:
    config = write_config(tmp_path)

    result = CliRunner().invoke(app, ["rate", "--rater", "r1", "--config", str(config)])

    assert result.exit_code == 0, result.output
    (command,) = launched
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4].endswith("app.py")
    assert command[5:9] == [
        "--server.address",
        "localhost",
        "--browser.gatherUsageStats",
        "false",
    ]
    assert command[9:] == ["--", "--config", str(config), "--rater", "r1", "--share", "1.0"]


def test_additional_raters_get_the_configured_overlap(
    tmp_path: Path, launched: list[list[str]]
) -> None:
    config = write_config(tmp_path)

    CliRunner().invoke(app, ["rate", "--rater", "r2", "--additional", "--config", str(config)])

    assert launched[0][-2:] == ["--share", "0.2"]


@pytest.mark.parametrize("rater", ["Carlos", "carlos@example.com"])
def test_refuses_names_that_are_not_pseudonyms(
    tmp_path: Path, launched: list[list[str]], rater: str
) -> None:
    result = CliRunner().invoke(
        app, ["rate", "--rater", rater, "--config", str(write_config(tmp_path))]
    )

    assert result.exit_code == 2
    assert "short lowercase pseudonym" in result.output
    assert launched == []


def test_explains_a_missing_app_extra(
    tmp_path: Path, launched: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    result = CliRunner().invoke(
        app, ["rate", "--rater", "r1", "--config", str(write_config(tmp_path))]
    )

    assert result.exit_code == 2
    assert "uv sync --extra app" in result.output


def test_reports_an_invalid_config(tmp_path: Path, launched: list[list[str]]) -> None:
    result = CliRunner().invoke(
        app, ["rate", "--rater", "r1", "--config", str(tmp_path / "nope.yaml")]
    )

    assert result.exit_code == 2
    assert "nope.yaml" in result.output
