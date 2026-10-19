import pytest
from pydantic import ValidationError, computed_field

from assay.records import Record


class Span(Record):
    start: int
    end: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def length(self) -> int:
        return self.end - self.start


def test_round_trips_with_computed_fields() -> None:
    span = Span(start=2, end=5)

    assert span.model_dump() == {"start": 2, "end": 5, "length": 3}
    assert Span.model_validate(span.model_dump()) == span
    assert Span.model_validate_json(span.model_dump_json()) == span


def test_accepts_an_existing_instance() -> None:
    span = Span(start=2, end=5)

    assert Span.model_validate(span) is span


def test_non_mapping_input_is_left_for_pydantic_to_reject() -> None:
    with pytest.raises(ValidationError, match="valid dictionary"):
        Span.model_validate([2, 5])


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Span.model_validate({"start": 1, "end": 2, "colour": "red"})


def test_is_immutable() -> None:
    with pytest.raises(ValidationError):
        Span(start=1, end=2).start = 0  # type: ignore[misc]
