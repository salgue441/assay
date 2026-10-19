"""A small record type and table spec shared by the storage tests."""

from typing import Self

from pydantic import Field, computed_field, model_validator

from assay.records import Record
from assay.storage import TableSpec


class Reading(Record):
    repo: str
    epoch: int = Field(ge=0)
    sensor: str
    low: int
    high: int
    note: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.high < self.low:
            msg = "high is below low"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> int:
        return self.high - self.low


READINGS = TableSpec(name="readings", model=Reading, key=("repo", "epoch", "sensor"))


def reading(
    sensor: str = "a", *, repo: str = "owner/repo", epoch: int = 0, **fields: object
) -> Reading:
    values: dict[str, object] = {"low": 1, "high": 3} | fields
    return Reading.model_validate({"repo": repo, "epoch": epoch, "sensor": sensor, **values})
