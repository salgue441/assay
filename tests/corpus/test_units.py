from pathlib import PurePosixPath
from textwrap import dedent

import pytest
from pydantic import ValidationError

from assay.corpus import SourceFile, Unit, UnitPolicy, changed_units, select_units
from assay.extract import PythonExtractor, SymbolKind

COMMIT = "c" * 40

SOURCE = dedent(
    '''
    """Payments module."""

    import os


    class Client:
        """Talks to the API."""

        def send(self) -> None:
            pass

        class Retry:
            pass


    class _Private:
        def hidden(self) -> None:
            pass


    def charge(amount: int) -> None:
        """Charge an amount."""
        print(amount)


    def tiny() -> None: ...
    '''
).lstrip("\n")


def units_for(source: str = SOURCE, *, policy: UnitPolicy | None = None) -> dict[str, Unit]:
    file = SourceFile(path=PurePosixPath("pay/core.py"), module="pay.core", text=source)
    inventory = PythonExtractor().extract(file.text, module=file.module, path=str(file.path))
    selected = select_units(
        inventory, file, repo="owner/pay", commit=COMMIT, policy=policy or UnitPolicy()
    )
    return {u.qualified_name: u for u in selected}


def test_selects_the_module_and_public_top_level_classes_and_functions() -> None:
    assert list(units_for()) == ["pay.core", "pay.core.Client", "pay.core.charge"]


def test_unit_source_is_exactly_the_symbol_lines() -> None:
    charge = units_for()["pay.core.charge"]

    assert charge.source == (
        'def charge(amount: int) -> None:\n    """Charge an amount."""\n    print(amount)\n'
    )
    assert (charge.start_line, charge.end_line, charge.size_lines) == (21, 23, 3)
    assert charge.kind is SymbolKind.FUNCTION


def test_module_unit_is_the_whole_file_and_contains_every_symbol() -> None:
    module = units_for()["pay.core"]

    assert module.source == SOURCE
    assert "pay.core._Private.hidden" in module.symbols
    assert len(module.symbols) == 8


def test_class_unit_contains_its_members_but_not_siblings() -> None:
    client = units_for()["pay.core.Client"]

    assert client.symbols == (
        "pay.core.Client",
        "pay.core.Client.send",
        "pay.core.Client.Retry",
    )


def test_identity_and_hash() -> None:
    client = units_for()["pay.core.Client"]

    assert client.unit_id == "owner/pay:pay/core.py#pay.core.Client"
    assert client.repo == "owner/pay"
    assert client.commit == COMMIT
    assert len(client.unit_hash) == 64


def test_policy_controls_kinds_size_and_privacy() -> None:
    policy = UnitPolicy(
        kinds=frozenset({SymbolKind.CLASS, SymbolKind.FUNCTION}), min_lines=1, include_private=True
    )

    assert list(units_for(policy=policy)) == [
        "pay.core.Client",
        "pay.core._Private",
        "pay.core.charge",
        "pay.core.tiny",
    ]


@pytest.mark.parametrize(
    "policy",
    [
        {"kinds": frozenset({SymbolKind.METHOD})},
        {"min_lines": 0},
    ],
)
def test_policy_rejects_invalid_settings(policy: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UnitPolicy(**policy)  # type: ignore[arg-type]


def test_unit_rejects_method_kind() -> None:
    fields = units_for()["pay.core.Client"].model_dump() | {"kind": SymbolKind.METHOD}

    with pytest.raises(ValidationError, match="not a unit kind"):
        Unit.model_validate(fields)


def test_unit_round_trips_through_json() -> None:
    unit = units_for()["pay.core.charge"]

    assert Unit.model_validate_json(unit.model_dump_json()) == unit


def test_changed_units_finds_new_and_modified_units_only() -> None:
    before = units_for()
    after = units_for(
        SOURCE.replace("print(amount)", "print(amount * 2)")
        + "\n\ndef refund() -> None:\n    pass\n    pass\n"
    )

    changed = {u.qualified_name for u in changed_units(before.values(), after.values())}

    assert changed == {"pay.core", "pay.core.charge", "pay.core.refund"}


def test_changed_units_with_no_history_returns_everything() -> None:
    units = units_for()

    assert len(changed_units([], units.values())) == len(units)
