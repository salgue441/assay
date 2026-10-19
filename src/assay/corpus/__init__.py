"""
Build the source corpus: pinned repositories, their source files and the
documentation units cut from them.
"""

from assay.corpus.builder import CorpusBuilder, SnapshotCorpus
from assay.corpus.discovery import SourceFile, discover_sources
from assay.corpus.layout import LanguageLayout, PythonLayout, get_layout
from assay.corpus.manifest import (
    CommitSha,
    DisallowedLicenseError,
    Manifest,
    ManifestError,
    RepoSpec,
    load_manifest,
)
from assay.corpus.snapshot import GitSnapshotFetcher, Snapshot, SnapshotError, SnapshotFetcher
from assay.corpus.units import Unit, UnitPolicy, changed_units, select_units

__all__ = [
    "CommitSha",
    "CorpusBuilder",
    "DisallowedLicenseError",
    "GitSnapshotFetcher",
    "LanguageLayout",
    "Manifest",
    "ManifestError",
    "PythonLayout",
    "RepoSpec",
    "Snapshot",
    "SnapshotCorpus",
    "SnapshotError",
    "SnapshotFetcher",
    "SourceFile",
    "Unit",
    "UnitPolicy",
    "changed_units",
    "discover_sources",
    "get_layout",
    "load_manifest",
    "select_units",
]
