"""
Build the source corpus: pinned repositories, their source files and the
documentation units cut from them.
"""

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
from assay.corpus.units import Unit, UnitPolicy, changed_units, select_units

__all__ = [
    "CommitSha",
    "DisallowedLicenseError",
    "LanguageLayout",
    "Manifest",
    "ManifestError",
    "PythonLayout",
    "RepoSpec",
    "SourceFile",
    "Unit",
    "UnitPolicy",
    "changed_units",
    "discover_sources",
    "get_layout",
    "load_manifest",
    "select_units",
]
