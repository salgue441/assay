from datetime import date, datetime
from enum import StrEnum

import pyarrow as pa
import pytest
from pydantic import AwareDatetime, BaseModel, Field, computed_field

from assay.storage import arrow_schema


class Colour(StrEnum):
    RED = "red"


class Everything(BaseModel):
    name: str
    count: int
    flag: bool
    ratio: float
    day: date
    at: AwareDatetime
    colour: Colour
    note: str | None
    tags: tuple[str, ...]
    scores: list[int | None]
    secret: str = Field(exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def loud_name(self) -> str:
        return self.name.upper()


def test_maps_every_supported_annotation() -> None:
    schema = arrow_schema(Everything)

    assert schema == pa.schema(
        [
            pa.field("name", pa.string(), nullable=False),
            pa.field("count", pa.int64(), nullable=False),
            pa.field("flag", pa.bool_(), nullable=False),
            pa.field("ratio", pa.float64(), nullable=False),
            pa.field("day", pa.date32(), nullable=False),
            pa.field("at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("colour", pa.string(), nullable=False),
            pa.field("note", pa.string(), nullable=True),
            pa.field(
                "tags", pa.list_(pa.field("item", pa.string(), nullable=False)), nullable=False
            ),
            pa.field(
                "scores", pa.list_(pa.field("item", pa.int64(), nullable=True)), nullable=False
            ),
            pa.field("loud_name", pa.string(), nullable=False),
        ]
    )


def test_excluded_fields_are_not_stored() -> None:
    assert "secret" not in arrow_schema(Everything).names


class Point(BaseModel):
    x: int
    label: str | None


class Shape(BaseModel):
    corners: tuple[Point, ...]
    centre: Point


def test_nested_models_become_structs() -> None:
    point = pa.struct(
        [pa.field("x", pa.int64(), nullable=False), pa.field("label", pa.string(), nullable=True)]
    )

    assert arrow_schema(Shape) == pa.schema(
        [
            pa.field("corners", pa.list_(pa.field("item", point, nullable=False)), nullable=False),
            pa.field("centre", point, nullable=False),
        ]
    )


def test_nested_models_round_trip_through_arrow() -> None:
    shape = Shape(
        corners=(Point(x=1, label="a"), Point(x=2, label=None)), centre=Point(x=0, label=None)
    )

    table = pa.Table.from_pylist([shape.model_dump()], schema=arrow_schema(Shape))

    assert Shape.model_validate(table.to_pylist()[0]) == shape


class NaiveTime(BaseModel):
    at: datetime


class Mapping(BaseModel):
    lookup: dict[str, int]


class FixedTuple(BaseModel):
    pair: tuple[str, int]


class EitherOr(BaseModel):
    value: int | str


@pytest.mark.parametrize("model", [NaiveTime, Mapping, FixedTuple, EitherOr])
def test_rejects_annotations_without_a_safe_arrow_type(model: type[BaseModel]) -> None:
    with pytest.raises(TypeError, match="no Arrow type"):
        arrow_schema(model)
