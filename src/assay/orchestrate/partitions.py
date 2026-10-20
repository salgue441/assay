"""
Dagster partitions for the corpus: one per repository epoch in the manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import dagster as dg

from assay.corpus import Manifest, RepoSpec
from assay.storage import repo_slug


def partition_key(repo: str, epoch: int) -> str:
    """`owner__repo@07`: sorts by repository, then epoch, in the Dagster UI."""

    return f"{repo_slug(repo)}@{epoch:02d}"


@dataclass(frozen=True)
class CorpusPartitions:
    """
    Every repository epoch in a manifest, keyed by partition key.

    Keys are looked up, never parsed, so the key format can change without
    touching the code that resolves them.
    """

    epochs: Mapping[str, tuple[RepoSpec, int]]

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> CorpusPartitions:
        return cls(
            {
                partition_key(repo.name, epoch): (repo, epoch)
                for repo in manifest.repos
                for epoch in range(len(repo.epochs))
            }
        )

    @cached_property
    def definition(self) -> dg.StaticPartitionsDefinition:
        return dg.StaticPartitionsDefinition(sorted(self.epochs))

    def resolve(self, key: str) -> tuple[RepoSpec, int]:
        try:
            return self.epochs[key]
        except KeyError:
            msg = f"unknown corpus partition {key!r}"
            raise KeyError(msg) from None

    def keys(self, *, repo: str | None = None, epoch: int | None = None) -> list[str]:
        """Partition keys, optionally limited to one repository and/or epoch."""

        return sorted(
            key
            for key, (spec, number) in self.epochs.items()
            if (repo is None or spec.name == repo) and (epoch is None or number == epoch)
        )
