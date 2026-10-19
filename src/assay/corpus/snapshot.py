"""
Check repositories out at pinned commits.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from pydantic import AwareDatetime, Field

from assay.corpus.manifest import CommitSha, RepoSpec
from assay.extract import SourceLanguage
from assay.records import Record


class SnapshotError(RuntimeError):
    """Raised when a repository cannot be checked out at a commit."""


class Snapshot(Record):
    """
    One repository at one epoch's commit: a `sources` row.

    `tree_hash` is git's content hash of the commit's tree, so two snapshots
    with the same tree hash have identical files.
    """

    repo: str
    epoch: int = Field(ge=0)
    commit: CommitSha
    tree_hash: str = Field(pattern=r"^[0-9a-f]{40}$")
    committed_at: AwareDatetime
    """When the commit was made: no code in the snapshot is newer, which dates
    units for the per-unit contamination check against model training cutoffs."""
    license: str
    language: SourceLanguage
    created_at: date


@dataclass(frozen=True, slots=True)
class Checkout:
    """
    A snapshot's files on this machine. The location is machine-specific, so
    it stays out of the stored `Snapshot`.
    """

    snapshot: Snapshot
    root: Path


class SnapshotFetcher(Protocol):
    def fetch(self, repo: RepoSpec, epoch: int) -> Checkout:
        """Make `repo` available locally at `repo.epochs[epoch]`."""
        ...


class GitSnapshotFetcher:
    """
    Shallow-fetch each commit into its own directory under `cache_dir`.

    Checkouts are reused when they are already at the right commit and clean,
    and are built in a `.partial` directory that is renamed into place only
    when complete, so an interrupted run never leaves a half-made snapshot.
    """

    def __init__(self, cache_dir: Path, *, git: str = "git", timeout_seconds: float = 600) -> None:
        self._cache_dir = cache_dir
        self._git_executable = git
        self._timeout_seconds = timeout_seconds

    def fetch(self, repo: RepoSpec, epoch: int) -> Checkout:
        if not 0 <= epoch < len(repo.epochs):
            msg = f"{repo.name} has {len(repo.epochs)} epochs; epoch {epoch} does not exist"
            raise IndexError(msg)

        commit = repo.epochs[epoch]
        target = self._cache_dir / repo.slug / commit
        if not self._is_clean_checkout(target, commit):
            self._checkout(repo.url, commit, target)

        snapshot = Snapshot(
            repo=repo.name,
            epoch=epoch,
            commit=commit,
            tree_hash=self._git(target, "rev-parse", "HEAD^{tree}"),
            committed_at=datetime.fromisoformat(
                self._git(target, "show", "--no-patch", "--format=%cI", "HEAD")
            ),
            license=repo.license,
            language=repo.language,
            created_at=repo.created_at,
        )
        return Checkout(snapshot=snapshot, root=target)

    def _is_clean_checkout(self, target: Path, commit: str) -> bool:
        if not (target / ".git").is_dir():
            return False
        try:
            head = self._git(target, "rev-parse", "HEAD")
            changes = self._git(target, "status", "--porcelain", "--untracked-files=all")
        except SnapshotError:
            return False
        return head == commit and not changes

    def _checkout(self, url: str, commit: str, target: Path) -> None:
        partial = target.with_name(f"{target.name}.partial")
        _remove_tree(partial)
        partial.mkdir(parents=True)

        self._git(partial, "init", "--quiet")
        self._git(partial, "fetch", "--quiet", "--depth", "1", "--no-tags", "--", url, commit)
        self._git(partial, "-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD")

        head = self._git(partial, "rev-parse", "HEAD")
        if head != commit:  # pragma: no cover - git guarantees this for a full SHA
            msg = f"{url}: fetched {head}, expected {commit}"
            raise SnapshotError(msg)

        _remove_tree(target)
        partial.rename(target)

    def _git(self, cwd: Path, *args: str) -> str:
        command = [self._git_executable, "-C", str(cwd), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            msg = f"git {args[0]} could not run: {exc}"
            raise SnapshotError(msg) from exc

        if result.returncode != 0:
            msg = f"git {' '.join(args[:2])} failed: {result.stderr.strip()}"
            raise SnapshotError(msg)
        return result.stdout.strip()


def _remove_tree(path: Path) -> None:
    """
    Delete a directory tree, including git's read-only object files on Windows.
    """

    def make_writable_and_retry(
        function: Callable[[str], object], failed: str, _: BaseException
    ) -> None:
        os.chmod(failed, stat.S_IWRITE)
        function(failed)

    if path.exists():
        shutil.rmtree(path, onexc=make_writable_and_retry)
