"""
Derive Arrow schemas from record models, so column types are declared once.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

import pyarrow as pa
from pydantic import AwareDatetime, BaseModel

_SCALARS: dict[Any, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    bool: pa.bool_(),
    float: pa.float64(),
    date: pa.date32(),
    AwareDatetime: pa.timestamp("us", tz="UTC"),
}


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """
    One column per serialized field, in declaration order, then one per
    computed field. Excluded fields are not stored.

    Raises `TypeError` for an annotation with no Arrow equivalent, including
    timezone-naive datetimes: stored timestamps must be unambiguous.
    """

    columns = [
        _field(name, info.annotation)
        for name, info in model.model_fields.items()
        if not info.exclude
    ]
    columns += [
        _field(name, info.return_type) for name, info in model.model_computed_fields.items()
    ]
    return pa.schema(columns)


def _field(name: str, annotation: Any) -> pa.Field:
    data_type, nullable = _arrow_type(annotation)
    return pa.field(name, data_type, nullable=nullable)


def _arrow_type(annotation: Any) -> tuple[pa.DataType, bool]:
    """
    The Arrow type for an annotation, and whether the column is nullable.
    """

    origin, args = get_origin(annotation), get_args(annotation)

    if origin in (Union, UnionType) and NoneType in args:
        members = [arg for arg in args if arg is not NoneType]
        if len(members) == 1:
            return _arrow_type(members[0])[0], True

    is_variadic_tuple = origin is tuple and len(args) == 2 and args[1] is Ellipsis
    if is_variadic_tuple or origin is list:
        item_type, item_nullable = _arrow_type(args[0])
        return pa.list_(pa.field("item", item_type, nullable=item_nullable)), False

    if (
        isinstance(annotation, type)
        and issubclass(annotation, Enum)
        and issubclass(annotation, str)
    ):
        return pa.string(), False

    if annotation in _SCALARS:
        return _SCALARS[annotation], False

    msg = f"no Arrow type for {annotation!r}"
    raise TypeError(msg)
