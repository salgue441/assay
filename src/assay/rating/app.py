"""
The rating app: one document at a time, blind and in the rater's own order.

Start it with `assay rate --rater <pseudonym>`. Raters see the source and the
document, never which model wrote it or how it was scored. Nothing is
preselected, so no default value can anchor a rating.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import streamlit as st

from assay.rating.ratings import Decision
from assay.rating.session import RatingItem, RatingSession
from assay.settings import load_settings
from assay.storage import ParquetStore

OPTIONS_KEY = "assay_options"
"""Session-state key holding the app's options; tests set it before running."""

SCORES = [1, 2, 3, 4, 5]
OVERALL_CAPTIONS = [
    "Unusable: wrong or not documentation of this code",
    "Poor: a reader would need the source to understand it",
    "Acceptable: right overall, with gaps or problems",
    "Good: correct and useful, with minor problems",
    "Excellent: correct, clear and complete",
]
DECISION_CAPTIONS = {
    Decision.PUBLISH: "Usable as is; no factual errors.",
    Decision.REVISE: "Has problems a reviewer could fix by editing.",
    Decision.ESCALATE: "Misleading or wrong; needs the source re-read or a new document.",
}


def _options() -> argparse.Namespace:
    if OPTIONS_KEY not in st.session_state:
        parser = argparse.ArgumentParser(prog="assay rate")
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--rater", required=True)
        parser.add_argument("--share", type=float, default=1.0)
        st.session_state[OPTIONS_KEY] = parser.parse_args(sys.argv[1:])
    options: argparse.Namespace = st.session_state[OPTIONS_KEY]
    return options


def _session() -> RatingSession:
    if "assay_session" not in st.session_state:
        options = _options()
        settings = load_settings(options.config)
        st.session_state["assay_session"] = RatingSession(
            ParquetStore(settings.storage.root),
            settings.judge.rubric(),
            rater_id=options.rater,
            seed=settings.run.seed,
            share=options.share,
        )
    session: RatingSession = st.session_state["assay_session"]
    return session


def _show_document(item: RatingItem) -> None:
    unit = item.unit
    st.subheader(f"{unit.kind.value} `{unit.qualified_name}`")
    source, document = st.columns(2)
    with source:
        st.caption(f"Source: {unit.path}, lines {unit.start_line}-{unit.end_line}")
        st.code(unit.source, language=item.language)
    with document:
        st.caption("Documentation to rate")
        st.markdown(item.document.text)  # HTML stays disabled: the text is untrusted
        with st.expander("Raw Markdown"):
            st.code(item.document.text, language="markdown")


def _rating_form(session: RatingSession, item: RatingItem) -> None:
    key = item.document.document_id[:16]
    with st.form(key=f"form-{key}"):
        scores = {
            name: st.radio(
                f"**{name.capitalize()}**: {criterion.question}",
                SCORES,
                index=None,
                captions=[criterion.levels[score] for score in SCORES],
                key=f"{name}-{key}",
            )
            for name, criterion in session.rubric.criteria.items()
        }
        overall = st.radio(
            "**Overall quality**",
            SCORES,
            index=None,
            captions=OVERALL_CAPTIONS,
            key=f"overall-{key}",
        )
        decision = st.radio(
            "**Decision**: what would you do with this document?",
            list(Decision),
            index=None,
            format_func=lambda d: d.value.capitalize(),
            captions=list(DECISION_CAPTIONS.values()),
            key=f"decision-{key}",
        )
        notes = st.text_area("Notes (optional)", key=f"notes-{key}")
        submitted = st.form_submit_button("Save and next", type="primary")

    if not submitted:
        return
    chosen = {name: score for name, score in scores.items() if score is not None}
    if overall is None or decision is None or len(chosen) < len(scores):
        missing = [name for name in scores if name not in chosen]
        missing += [
            label
            for label, value in (("overall", overall), ("decision", decision))
            if value is None
        ]
        st.error(f"Choose a value for: {', '.join(missing)}.")
        return

    session.submit(
        item,
        scores=chosen,
        overall=overall,
        decision=decision,
        notes=notes,
        seconds_spent=time.monotonic() - st.session_state["started_at"],
    )
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Assay rating", layout="wide")
    session = _session()
    done, total = session.progress

    with st.sidebar:
        st.markdown(f"**Rater** `{session.rater_id}`")
        st.progress(done / total if total else 1.0, text=f"{done} of {total} rated")
        st.caption(
            f"Rubric {session.rubric.version}. Which model wrote each document, and how "
            "it was scored, are hidden on purpose. Rate what you see."
        )

    item = session.next_item()
    if item is None:
        st.success(f"All {total} assigned documents are rated. Thank you.")
        return

    if st.session_state.get("shown") != item.document.document_id:
        st.session_state["shown"] = item.document.document_id
        st.session_state["started_at"] = time.monotonic()
    _show_document(item)
    _rating_form(session, item)


main()
