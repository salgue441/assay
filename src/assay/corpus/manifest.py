"""
The corpus manifest: which repositories enter the corpus, at which commits.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import Field, StringConstraints, ValidationError, model_validator

from assay.extract import SourceLanguage
from assay.records import Record

CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
"""A full, lowercase 40-character commit SHA; abbreviated SHAs are ambiguous."""


class ManifestError(ValueError):
    """Raised when the manifest cannot be read or is invalid."""


class DisallowedLicenseError(ValueError):
    """Raised when a repository's license is not on the allowlist."""


class RepoSpec(Record):
    """
    One repository and the ordered commits (epochs) it contributes.

    The first epoch is the calibration baseline; each later one is a drift
    epoch.
    """

    url: str = Field(pattern=r"^(https|file)://\S+$")
    license: str = Field(min_length=1)
    language: SourceLanguage
    created_at: date
    epochs: tuple[CommitSha, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_epochs(self) -> Self:
        if len(set(self.epochs)) != len(self.epochs):
            msg = f"{self.url}: epochs must not repeat a commit"
            raise ValueError(msg)
        return self

    @property
    def name(self) -> str:
        """
        `owner/repo`, from the last two segments of the URL.
        """

        segments = self.url.rstrip("/").removesuffix(".git").split("/")
        return "/".join(segments[-2:])

    @property
    def slug(self) -> str:
        """
        `name` made safe for a single directory name.
        """

        return self.name.replace("/", "__")


class Manifest(Record):
    repos: tuple[RepoSpec, ...] = ()

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        names = [repo.name for repo in self.repos]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"repositories listed more than once: {', '.join(duplicates)}"
            raise ValueError(msg)
        return self

    def check_licenses(self, allowed: Collection[str]) -> None:
        """
        Raise `DisallowedLicenseError` naming every repository whose license
        is not in `allowed`.
        """

        rejected = [f"{r.name} ({r.license})" for r in self.repos if r.license not in allowed]
        if rejected:
            msg = f"licenses not on the allowlist: {', '.join(rejected)}"
            raise DisallowedLicenseError(msg)


def load_manifest(path: Path) -> Manifest:
    """
    Read and validate a manifest file; any problem raises `ManifestError`
    naming the file.
    """

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(raw or {})
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        msg = f"{path}: {exc}"
        raise ManifestError(msg) from exc
