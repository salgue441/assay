import pytest
from pydantic import ValidationError

from assay.extract import SourceLanguage, Symbol, SymbolInventory, SymbolKind


def make_symbol(qualified_name: str = "pkg.func", **overrides: object) -> Symbol:
    fields: dict[str, object] = {
        "qualified_name": qualified_name,
        "name": qualified_name.rsplit(".", 1)[-1],
        "kind": SymbolKind.FUNCTION,
        "signature": None,
        "docstring": None,
        "start_line": 1,
        "end_line": 2,
        "is_public": True,
    }
    return Symbol.model_validate(fields | overrides)


def make_inventory(*symbols: Symbol) -> SymbolInventory:
    return SymbolInventory(
        module="pkg",
        path="pkg/__init__.py",
        language=SourceLanguage.PYTHON,
        has_syntax_errors=False,
        symbols=symbols,
    )


@pytest.mark.parametrize("public", [True, False])
def test_is_public_is_stored_as_given(public: bool) -> None:
    symbol = make_symbol("pkg._helper", is_public=public)

    assert symbol.is_public is public
    assert symbol.model_dump()["is_public"] is public


def test_is_public_is_required() -> None:
    fields = make_symbol().model_dump(exclude={"is_public"})

    with pytest.raises(ValidationError, match="is_public"):
        Symbol.model_validate(fields)


def test_rejects_end_line_before_start_line() -> None:
    with pytest.raises(ValidationError, match="before start_line"):
        make_symbol(start_line=5, end_line=4)


def test_rejects_zero_based_lines() -> None:
    with pytest.raises(ValidationError):
        make_symbol(start_line=0)


def test_rejects_qualified_name_that_does_not_end_with_name() -> None:
    with pytest.raises(ValidationError, match="does not end with name"):
        make_symbol("pkg.func", name="other")


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_symbol(colour="blue")


def test_symbols_are_immutable() -> None:
    with pytest.raises(ValidationError):
        make_symbol().name = "changed"  # type: ignore[misc]


def test_inventory_rejects_duplicate_qualified_names() -> None:
    with pytest.raises(ValidationError, match="duplicate qualified_name"):
        make_inventory(make_symbol("pkg.f"), make_symbol("pkg.f"))


def test_names_include_simple_and_qualified_forms() -> None:
    inventory = make_inventory(make_symbol("pkg.Client"), make_symbol("pkg.Client.send"))

    assert inventory.names() == {"Client", "pkg.Client", "send", "pkg.Client.send"}


def test_public_filters_private_symbols() -> None:
    inventory = make_inventory(make_symbol("pkg.run"), make_symbol("pkg.internal", is_public=False))

    assert [s.name for s in inventory.public()] == ["run"]


def test_inventory_round_trips_through_json() -> None:
    inventory = make_inventory(make_symbol("pkg.run", docstring="Run it."))

    assert SymbolInventory.model_validate_json(inventory.model_dump_json()) == inventory
