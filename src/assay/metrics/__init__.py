"""
Deterministic metrics that check generated documents against their source.
"""

from assay.metrics.base import Metric, MetricResult
from assay.metrics.context import MetricContext
from assay.metrics.coverage import SymbolCoverage, expected_names
from assay.metrics.integrity import ReferentialIntegrity, code_references
from assay.metrics.markdown import Heading, MarkdownParts, parse_markdown
from assay.metrics.measure import METRICS_TABLE, MetricRow, measure, static_metrics
from assay.metrics.readability import Readability
from assay.metrics.rules import LanguageRules, PythonRules, get_rules
from assay.metrics.structure import StructuralCompleteness, required_sections

__all__ = [
    "METRICS_TABLE",
    "Heading",
    "LanguageRules",
    "MarkdownParts",
    "Metric",
    "MetricContext",
    "MetricResult",
    "MetricRow",
    "PythonRules",
    "Readability",
    "ReferentialIntegrity",
    "StructuralCompleteness",
    "SymbolCoverage",
    "code_references",
    "expected_names",
    "get_rules",
    "measure",
    "parse_markdown",
    "required_sections",
    "static_metrics",
]
