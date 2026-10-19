"""
The contract every language extractor implements.
"""

from __future__ import annotations

from typing import Protocol

from assay.extract.models import SourceLanguage, SymbolInventory


class SymbolExtractor(Protocol):
    """
    Turns the source of one file into its symbol inventory.

    Implementations are deterministic: the same source, module and path always
    produce the same inventory.
    """

    @property
    def language(self) -> SourceLanguage: ...

    def extract(self, source: str, *, module: str, path: str = "") -> SymbolInventory:
        """
        Extract the symbols of one file.

        `module` is the dotted import name of the file (for example
        `package.sub.module`) and prefixes every qualified name; `path` is
        recorded for lineage only.
        """
        ...
