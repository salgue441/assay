from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from ..local_repo import LocalRepo
from .project import Project

PROJECT_ROOT = Path(__file__).parents[2]

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
                "generation": {
                    "prompt_version": "v1",
                    "prompt_dir": str(PROJECT_ROOT / "config" / "prompts" / "generation"),
                    "subjects": [{"name": "writer", "provider": "ollama", "model_id": "fake:1b"}],
                },
                "judge": {
                    "rubric_version": "v1",
                    "rubric_dir": str(PROJECT_ROOT / "config" / "rubrics"),
                    "runs_per_criterion": 2,
                    "criteria": ["accuracy", "clarity"],
                    "variance_study": {"temperatures": [0.0], "subset_size": 1},
                    "models": [{"name": "grader", "provider": "ollama", "model_id": "fake:1b"}],
                },
                "scoring": {
                    "weights": {"referential_integrity": 1.0},
                    "judge": "grader",
                    "critical_failures": {"referential_integrity_below": 0.9},
                    "routing": {"publish_at_or_above": 80, "escalate_below": 50},
                },
            }
        ),
        encoding="utf-8",
    )
    return Project(config=config, repo=repo, data=tmp_path / "data")
