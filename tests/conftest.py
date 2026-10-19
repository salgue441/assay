from collections.abc import Callable
from pathlib import Path

import pytest

from .local_repo import LocalRepo


@pytest.fixture
def make_repo(tmp_path: Path) -> Callable[[str], LocalRepo]:
    def factory(name: str = "owner/project") -> LocalRepo:
        return LocalRepo.init(tmp_path / "remotes" / name)

    return factory
