from collections.abc import Callable
from pathlib import Path

from assay.corpus import CorpusBuilder, GitSnapshotFetcher, UnitPolicy, changed_units

from .local_repo import LocalRepo

MakeRepo = Callable[..., LocalRepo]

CORE_V1 = '''"""Core module."""


def greet(name: str) -> str:
    """Say hello."""
    return f"hello {name}"


def wave() -> None:
    """Wave."""
    pass
'''


def test_builds_inventories_and_units_for_each_epoch(make_repo: MakeRepo, tmp_path: Path) -> None:
    repo = make_repo("owner/greeter")
    repo.commit(
        {
            "src/greeter/__init__.py": "",
            "src/greeter/core.py": CORE_V1,
            "tests/test_core.py": "def test(): ...\n",
        }
    )
    repo.commit({"src/greeter/core.py": CORE_V1.replace('f"hello {name}"', 'f"hi {name}"')})
    builder = CorpusBuilder(GitSnapshotFetcher(tmp_path / "cache"), policy=UnitPolicy(min_lines=2))

    first = builder.build(repo.spec(), 0)
    second = builder.build(repo.spec(), 1)

    assert [i.module for i in first.inventories] == ["greeter", "greeter.core"]
    assert [u.qualified_name for u in first.units] == [
        "greeter.core",
        "greeter.core.greet",
        "greeter.core.wave",
    ]
    assert first.units[1].commit == repo.commits[0]
    assert first.units[1].path == "src/greeter/core.py"

    changed = changed_units(first.units, second.units)
    assert [u.qualified_name for u in changed] == ["greeter.core", "greeter.core.greet"]
