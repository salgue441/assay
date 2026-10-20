from assay.metrics import MetricContext, PythonRules

from .fixtures import SYMBOLS, UNITS, context


def test_the_unit_symbol_comes_first() -> None:
    wrapper = context("Wrapper")

    assert wrapper.unit_symbols[0].qualified_name == "textkit.core.Wrapper"
    assert wrapper.symbol is wrapper.unit_symbols[0]


def test_known_names_include_source_identifiers_and_symbol_parts() -> None:
    known = context("log").known_names

    assert {"message", "print", "log", "textkit", "core", "textkit.core.Wrapper.wrap"} <= known


def test_facts_degrade_gracefully_without_stored_symbols() -> None:
    bare = MetricContext.build(UNITS["wrap_text"], (), PythonRules())

    assert bare.symbol is None
    assert (bare.parameters, bare.return_annotation, bare.public_members) == ((), None, ())


def test_a_class_without_a_constructor_has_no_constructor_parameters() -> None:
    no_init = [s for s in SYMBOLS if not s.qualified_name.endswith("__init__")]

    assert (
        MetricContext.build(UNITS["Wrapper"], no_init, PythonRules()).constructor_parameters == ()
    )


def test_only_functions_have_parameters() -> None:
    assert context("Wrapper").parameters == ()
    assert context("module").constructor_parameters == ()
