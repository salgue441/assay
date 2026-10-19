from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from ..local_repo import LocalRepo
from .project import Project

MODULE = '''"""Greetings."""


def greet(name: str) -> str:
    """Say hello."""
    return f"hello {name}"
'''


@pytest.fixture
def project(tmp_path: Path, make_repo: Callable[[str], LocalRepo]) -> Project:
    repo = make_repo("owner/greeter")
    repo.commit({"greeter/__init__.py": "", "greeter/core.py": MODULE})
    repo.commit({"greeter/core.py": MODULE.replace("hello", "hi")})

    manifest = tmp_path / "repos.yaml"
    manifest.write_text(
        yaml.safe_dump({"repos": [repo.spec().model_dump(mode="json")]}), encoding="utf-8"
    )

    # Absolute paths, so the config works from any working directory.
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run": {"seed": 1},
                "storage": {
                    "root": str(tmp_path / "data"),
                    "duckdb_path": str(tmp_path / "data" / "assay.duckdb"),
                },
                "corpus": {
                    "manifest": str(manifest),
                    "cache_dir": str(tmp_path / "cache"),
                    "allowed_licenses": ["MIT"],
                    "languages": ["python"],
                    "units": {"min_lines": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    return Project(config=config, repo=repo, data=tmp_path / "data")
