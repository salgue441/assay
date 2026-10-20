"""A throwaway project: a config file whose manifest points at a local repository."""

from dataclasses import dataclass
from pathlib import Path

from ..local_repo import LocalRepo


@dataclass
class Project:
    """A config file whose manifest points at a local git repository."""

    config: Path
    repo: LocalRepo
    data: Path
