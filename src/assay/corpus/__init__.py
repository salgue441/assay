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
from assay.corpus.snapshot import (
    Checkout,
    GitSnapshotFetcher,
    Snapshot,
    SnapshotError,
    SnapshotFetcher,
)
from assay.corpus.tables import (
    SOURCES,
    SYMBOLS,
    UNITS,
    SymbolRow,
    store_snapshot_corpus,
    symbol_rows,
)
from assay.corpus.units import Unit, UnitPolicy, changed_units, select_units

__all__ = [
    "SOURCES",
    "SYMBOLS",
    "UNITS",
    "Checkout",
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
    "SymbolRow",
    "Unit",
    "UnitPolicy",
    "changed_units",
    "discover_sources",
    "get_layout",
    "load_manifest",
    "select_units",
    "store_snapshot_corpus",
    "symbol_rows",
]
