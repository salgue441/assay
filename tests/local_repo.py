import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from assay.corpus import RepoSpec
from assay.corpus.snapshot import isolated_git_env
from assay.extract import SourceLanguage

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


@dataclass
class LocalRepo:
    """A real git repository in a temp directory, for snapshot tests."""

    path: Path
    commits: list[str]

    @classmethod
    def init(cls, path: Path) -> "LocalRepo":
        path.mkdir(parents=True)
        repo = cls(path=path, commits=[])
        repo._git("init", "--quiet", "--initial-branch=main")
        return repo

    @property
    def url(self) -> str:
        return self.path.as_uri()

    def commit(
        self, files: Mapping[str, str | None], message: str = "change", *, at: str | None = None
    ) -> str:
        """
        Write (or, with None, delete) files and commit; returns the new SHA.

        `at` is an ISO 8601 timestamp for both author and committer dates.
        """

        for relative, content in files.items():
            target = self.path / relative
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")

        dates = {"GIT_AUTHOR_DATE": at, "GIT_COMMITTER_DATE": at} if at else {}
        self._git("add", "--all")
        self._git("commit", "--quiet", "--allow-empty", "-m", message, extra_env=dates)
        sha = self._git("rev-parse", "HEAD")
        self.commits.append(sha)
        return sha

    def spec(self, *, license: str = "MIT") -> RepoSpec:
        return RepoSpec(
            url=self.url,
            license=license,
            language=SourceLanguage.PYTHON,
            created_at=date(2025, 1, 1),
            epochs=tuple(self.commits),
        )

    def _git(self, *args: str, extra_env: Mapping[str, str] | None = None) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True,
            text=True,
            env={**isolated_git_env(), **_GIT_IDENTITY, **(extra_env or {})},
            check=True,
        )
        return result.stdout.strip()
