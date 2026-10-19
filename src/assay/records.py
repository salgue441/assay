"""
Base class for the records Assay stores as table rows.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Record(BaseModel):
    """
    An immutable, strictly validated row.

    Unknown fields are rejected so typos fail loudly. Computed fields are
    written on serialization and dropped again on reload, so a stored row
    always validates back into the same record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in cls.model_computed_fields}
        return data
