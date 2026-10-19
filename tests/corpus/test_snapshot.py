from collections.abc import Callable
from pathlib import Path

import pytest

from assay.corpus import GitSnapshotFetcher, SnapshotError

from .local_repo import LocalRepo

MakeRepo = Callable[..., LocalRepo]


@pytest.fixture
def repo(make_repo: MakeRepo) -> LocalRepo:
    repo = make_repo("owner/project")
    repo.commit({"pkg/mod.py": "x = 1\n"}, "first")
    repo.commit({"pkg/mod.py": "x = 2\n"}, "second")
    return repo


def test_checks_out_each_epoch_at_its_commit(repo: LocalRepo, tmp_path: Path) -> None:
    fetcher = GitSnapshotFetcher(tmp_path / "cache")
    spec = repo.spec()

    first = fetcher.fetch(spec, 0)
    second = fetcher.fetch(spec, 1)

    assert (first.root / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (second.root / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 2\n"
    assert first.root == tmp_path / "cache" / "owner__project" / repo.commits[0]
    assert (first.commit, first.epoch, first.repo) == (repo.commits[0], 0, "owner/project")
    assert first.tree_hash != second.tree_hash


def test_snapshot_records_source_metadata(repo: LocalRepo, tmp_path: Path) -> None:
    snapshot = GitSnapshotFetcher(tmp_path / "cache").fetch(repo.spec(license="Apache-2.0"), 0)

    row = snapshot.model_dump()
    assert row["license"] == "Apache-2.0"
    assert "root" not in row


def test_reuses_a_clean_checkout(repo: LocalRepo, tmp_path: Path) -> None:
    fetcher = GitSnapshotFetcher(tmp_path / "cache")
    root = fetcher.fetch(repo.spec(), 0).root
    marker = root / ".git" / "assay-marker"
    marker.write_text("kept", encoding="utf-8")

    fetcher.fetch(repo.spec(), 0)

    assert marker.exists()


def test_rebuilds_a_modified_checkout(repo: LocalRepo, tmp_path: Path) -> None:
    fetcher = GitSnapshotFetcher(tmp_path / "cache")
    root = fetcher.fetch(repo.spec(), 0).root
    (root / "pkg" / "mod.py").write_text("tampered\n", encoding="utf-8")
    (root / "extra.py").write_text("", encoding="utf-8")

    fetcher.fetch(repo.spec(), 0)

    assert (root / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (root / "extra.py").exists()


def test_rebuilds_a_corrupted_checkout(repo: LocalRepo, tmp_path: Path) -> None:
    fetcher = GitSnapshotFetcher(tmp_path / "cache")
    root = fetcher.fetch(repo.spec(), 0).root
    (root / ".git" / "HEAD").write_text("garbage\n", encoding="utf-8")

    snapshot = fetcher.fetch(repo.spec(), 0)

    assert snapshot.commit == repo.commits[0]
    assert (root / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_replaces_a_leftover_partial_checkout(repo: LocalRepo, tmp_path: Path) -> None:
    partial = tmp_path / "cache" / "owner__project" / f"{repo.commits[0]}.partial"
    (partial / "junk").mkdir(parents=True)

    snapshot = GitSnapshotFetcher(tmp_path / "cache").fetch(repo.spec(), 0)

    assert not partial.exists()
    assert (snapshot.root / "pkg" / "mod.py").exists()


def test_unknown_commit_raises_snapshot_error(repo: LocalRepo, tmp_path: Path) -> None:
    spec = repo.spec().model_copy(update={"epochs": ("f" * 40,)})

    with pytest.raises(SnapshotError, match="git fetch"):
        GitSnapshotFetcher(tmp_path / "cache").fetch(spec, 0)


def test_out_of_range_epoch_is_rejected(repo: LocalRepo, tmp_path: Path) -> None:
    with pytest.raises(IndexError, match="has 2 epochs; epoch 2"):
        GitSnapshotFetcher(tmp_path / "cache").fetch(repo.spec(), 2)


def test_missing_git_executable_raises_snapshot_error(repo: LocalRepo, tmp_path: Path) -> None:
    fetcher = GitSnapshotFetcher(tmp_path / "cache", git="definitely-not-git")

    with pytest.raises(SnapshotError, match="could not run"):
        fetcher.fetch(repo.spec(), 0)
