"""
Readability of a document's prose.
"""

from __future__ import annotations

import textstat

from assay.metrics.base import MetricResult
from assay.metrics.context import MetricContext
from assay.metrics.markdown import parse_markdown

MIN_WORDS = 10
EASY_GRADE = 8.0
HARD_GRADE = 18.0


class Readability:
    """
    The Flesch-Kincaid grade of the prose, with code and headings removed,
    mapped linearly to [0, 1]: grade 8 or below scores 1.0, grade 18 or above
    scores 0.0. Fewer than 10 words of prose scores 0.0, as there is too
    little explanation to read.
    """

    name = "readability"
    version = "v1"

    def score(self, text: str, context: MetricContext) -> MetricResult:
        prose = parse_markdown(text).prose
        words = textstat.lexicon_count(prose)
        if words < MIN_WORDS:
            return MetricResult(value=0.0, explanation=f"too little prose to read ({words} words)")

        grade = textstat.flesch_kincaid_grade(prose)
        value = min(1.0, max(0.0, (HARD_GRADE - grade) / (HARD_GRADE - EASY_GRADE)))
        return MetricResult(
            value=value, explanation=f"Flesch-Kincaid grade {grade:.1f} over {words} words"
        )
