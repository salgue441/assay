"""
Deterministic symbol extraction: the ground truth for static metrics.
"""

from assay.extract.base import SymbolExtractor
from assay.extract.go import GoExtractor
from assay.extract.models import SourceLanguage, Symbol, SymbolInventory, SymbolKind
from assay.extract.python import PythonExtractor
from assay.extract.registry import UnsupportedLanguageError, get_extractor, supported_languages
from assay.extract.typescript import TypeScriptExtractor

__all__ = [
    "GoExtractor",
    "PythonExtractor",
    "SourceLanguage",
    "Symbol",
    "SymbolExtractor",
    "SymbolInventory",
    "SymbolKind",
    "TypeScriptExtractor",
    "UnsupportedLanguageError",
    "get_extractor",
    "supported_languages",
]
