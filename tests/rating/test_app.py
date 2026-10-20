from argparse import Namespace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import assay.rating
from assay.rating import RATINGS, Decision
from assay.storage import ParquetStore

from .fixtures import SUBJECTS, make_store, write_config

APP = Path(assay.rating.__file__).with_name("app.py")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    make_store(tmp_path / "data")
    return write_config(tmp_path)


def start(config: Path, rater: str = "r1", share: float = 1.0) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["assay_options"] = Namespace(config=config, rater=rater, share=share)
    return app.run()


def page_text(app: AppTest) -> str:
    parts = [e.value for e in (*app.markdown, *app.code, *app.subheader, *app.caption)]
    return "\n".join(str(p) for p in parts)


def fill(
    app: AppTest, *, overall: int | None = 4, decision: Decision | None = Decision.PUBLISH
) -> None:
    for radio in app.radio:
        if radio.key.startswith(("accuracy-", "clarity-", "consistency-")):
            radio.set_value(4)
    if overall is not None:
        next(r for r in app.radio if r.key.startswith("overall-")).set_value(overall)
    if decision is not None:
        next(r for r in app.radio if r.key.startswith("decision-")).set_value(decision)


def test_shows_the_source_and_document_without_revealing_the_model(config: Path) -> None:
    app = start(config)

    assert not app.exception
    text = page_text(app)
    assert "shapes.core" in text
    assert "## Doc" in text
    for subject in SUBJECTS:
        assert subject not in text, f"the subject model {subject!r} must stay hidden"


def test_nothing_is_preselected(config: Path) -> None:
    app = start(config)

    assert app.radio, "the form shows the rating scales"
    assert all(radio.value is None for radio in app.radio)


def test_an_incomplete_form_is_refused(config: Path, tmp_path: Path) -> None:
    app = start(config)
    fill(app, overall=None, decision=None)

    app.button[0].click().run()

    assert "Choose a value for: overall, decision." in app.error[0].value
    assert ParquetStore(tmp_path / "data").read(RATINGS) == []


def test_a_complete_form_saves_the_rating_and_shows_the_next_document(
    config: Path, tmp_path: Path
) -> None:
    app = start(config)
    first = app.subheader[0].value
    fill(app, overall=3, decision=Decision.REVISE)

    app.button[0].click().run()

    (rating,) = ParquetStore(tmp_path / "data").read(RATINGS)
    assert (rating.overall, rating.decision, rating.rater_id) == (3, Decision.REVISE, "r1")
    assert rating.seconds_spent >= 0
    assert app.get("progress")[0].proto.text == "1 of 10 rated"
    assert app.subheader[0].value != first


def test_finishing_shows_a_thank_you(config: Path) -> None:
    app = start(config, rater="r9", share=0.05)
    for _ in range(10):
        if app.success:
            break
        fill(app)
        app.button[0].click().run()

    assert "assigned documents are rated" in app.success[0].value
