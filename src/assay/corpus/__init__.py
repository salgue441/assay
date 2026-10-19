"""
Build the source corpus: pinned repositories, their source files and the
documentation units cut from them.
"""

from assay.corpus.manifest import (
    CommitSha,
    DisallowedLicenseError,
    Manifest,
    ManifestError,
    RepoSpec,
    load_manifest,
)

__all__ = [
    "CommitSha",
    "DisallowedLicenseError",
    "Manifest",
    "ManifestError",
    "RepoSpec",
    "load_manifest",
]
