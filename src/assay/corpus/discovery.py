"""
Find the source files in a snapshot.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from assay.corpus.layout import LanguageLayout

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "benchmarks",
        "build",
        "dist",
        "doc",
        "docs",
        "example",
        "examples",
        "migrations",
        "node_modules",
        "scripts",
        "site-packages",
        "test",
        "testing",
        "tests",
        "third_party",
        "vendor",
        "vendored",
    }
)
"""Directories that hold tests, examples, generated or third-party code."""

DEFAULT_MAX_FILE_BYTES = 1_000_000
"""Larger files are almost always generated, not hand-written source."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: PurePosixPath
    """Relative to the snapshot root."""
    module: str
    text: str


def discover_sources(
    root: Path,
    layout: LanguageLayout,
    *,
    excluded_dirs: Collection[str] = DEFAULT_EXCLUDED_DIRS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Iterator[SourceFile]:
    """
    Yield the source files under `root` in a deterministic (sorted) order.

    Hidden and excluded directories are not entered. Symlinks, oversized
    files and files that are not valid UTF-8 are skipped, as are files the
    layout cannot name. Line endings are normalized to `\\n`.
    """

    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith(".") and d not in excluded_dirs)

        for filename in sorted(filenames):
            absolute = Path(directory, filename)
            relative = PurePosixPath(absolute.relative_to(root).as_posix())
            if not layout.is_source_file(relative):
                continue

            source = _read_source(absolute, relative, layout, max_file_bytes)
            if source is not None:
                yield source


def _read_source(
    absolute: Path, relative: PurePosixPath, layout: LanguageLayout, max_file_bytes: int
) -> SourceFile | None:
    module = layout.module_name(relative)
    if module is None:
        logger.debug("skipping %s: no valid module name", relative)
        return None

    if absolute.is_symlink():
        logger.debug("skipping %s: symlink", relative)
        return None

    if absolute.stat().st_size > max_file_bytes:
        logger.debug("skipping %s: larger than %d bytes", relative, max_file_bytes)
        return None

    try:
        text = absolute.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("skipping %s: not valid UTF-8", relative)
        return None

    return SourceFile(path=relative, module=module, text=text)
