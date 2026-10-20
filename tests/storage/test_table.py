import pyarrow as pa
import pytest
from pydantic import Field

from assay.records import Record
from assay.storage import ContractError, TableSpec

from .records import READINGS, Reading, reading


def test_round_trips_records_through_arrow() -> None:
    rows = [reading("a"), reading("b", note="calibrated")]

    table = READINGS.to_arrow(rows)

    assert table.schema == READINGS.schema
    assert READINGS.validate(table) == rows


def test_valid_table_passes_the_contract() -> None:
    READINGS.validate(READINGS.to_arrow([reading("a"), reading("b")]))


def test_empty_table_passes_the_contract() -> None:
    READINGS.validate(READINGS.schema.empty_table())


def test_null_optional_values_pass_the_contract() -> None:
    # Regression: validating through pandas turned null strings into NaN.
    rows = [reading("a", note=None)]

    assert READINGS.validate(READINGS.to_arrow(rows)) == rows


def test_contract_rejects_duplicate_keys() -> None:
    table = READINGS.to_arrow([reading("a"), reading("a", high=9)])

    with pytest.raises(ContractError, match=r"key \{.*'sensor': 'a'\} appears 2 times"):
        READINGS.validate(table)


def test_contract_reports_every_problem_up_to_a_limit() -> None:
    rows = [reading(str(i)).model_dump() | {"low": 9, "high": 1} for i in range(25)]
    table = pa.Table.from_pylist(rows, schema=READINGS.schema)

    with pytest.raises(ContractError) as excinfo:
        READINGS.validate(table)

    message = str(excinfo.value)
    assert message.count("high is below low") == 20
    assert message.endswith("... and 5 more")


def test_contract_applies_the_models_own_validators() -> None:
    row = reading("a").model_dump() | {"low": 5, "high": 1, "spread": -4}
    table = pa.Table.from_pylist([row], schema=READINGS.schema)

    with pytest.raises(ContractError, match="high is below low"):
        READINGS.validate(table)


def test_contract_rejects_a_different_schema() -> None:
    table = READINGS.to_arrow([reading("a")]).drop_columns(["note"])

    with pytest.raises(ContractError, match="schema differs from Reading"):
        READINGS.validate(table)


class WithLocalPath(Record):
    name: str
    local_path: str = Field(exclude=True)


def test_rejects_models_whose_stored_rows_cannot_be_loaded_back() -> None:
    with pytest.raises(ValueError, match=r"required fields that are never stored \(local_path\)"):
        TableSpec(name="bad", model=WithLocalPath, key=("name",))


@pytest.mark.parametrize("key", [(), ("sensor", "missing")])
def test_key_must_be_existing_columns(key: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="must be non-empty columns of Reading"):
        TableSpec(name="bad", model=Reading, key=key)
