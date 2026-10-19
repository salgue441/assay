from typer.testing import CliRunner

from assay import __version__
from assay.cli import app


def test_version_command_prints_installed_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
