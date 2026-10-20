"""
Language knowledge the static checks need, behind one interface.

Adding a language means implementing `LanguageRules` and adding one entry to
`_RULES`; the metrics themselves do not change.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from assay.extract import SourceLanguage, UnsupportedLanguageError

_RECEIVERS = frozenset({"self", "cls"})


class LanguageRules(Protocol):
    @property
    def ambient_names(self) -> frozenset[str]:
        """Names any code can use without defining them: keywords, builtins,
        standard library modules and the attributes of builtin types."""
        ...

    def parameters(self, signature: str) -> tuple[str, ...]:
        """Parameter names of a function signature, excluding the receiver."""
        ...

    def return_annotation(self, signature: str) -> str | None:
        """The declared return type of a function signature, if any."""
        ...


def _python_ambient_names() -> frozenset[str]:
    builtin_types = [value for value in vars(builtins).values() if isinstance(value, type)]
    return frozenset(
        {
            *keyword.kwlist,
            *keyword.softkwlist,
            *_RECEIVERS,
            *dir(builtins),
            *sys.stdlib_module_names,
            *(attribute for kind in builtin_types for attribute in dir(kind)),
        }
    )


_PYTHON_AMBIENT_NAMES = _python_ambient_names()


class PythonRules:
    @property
    def ambient_names(self) -> frozenset[str]:
        return _PYTHON_AMBIENT_NAMES

    def parameters(self, signature: str) -> tuple[str, ...]:
        function = _parse_function(signature)
        if function is None:
            return ()
        arguments = function.args
        names = [
            *(a.arg for a in arguments.posonlyargs),
            *(a.arg for a in arguments.args),
            *([arguments.vararg.arg] if arguments.vararg else []),
            *(a.arg for a in arguments.kwonlyargs),
            *([arguments.kwarg.arg] if arguments.kwarg else []),
        ]
        if names and names[0] in _RECEIVERS:
            names = names[1:]
        return tuple(names)

    def return_annotation(self, signature: str) -> str | None:
        function = _parse_function(signature)
        if function is None or function.returns is None:
            return None
        return ast.unparse(function.returns)


def _parse_function(signature: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        module = ast.parse(f"{signature}:\n    pass")
    except SyntaxError:
        return None
    node = module.body[0] if module.body else None
    return node if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) else None


_RULES: Mapping[SourceLanguage, LanguageRules] = MappingProxyType(
    {
        SourceLanguage.PYTHON: PythonRules(),
    }
)


def get_rules(language: SourceLanguage | str) -> LanguageRules:
    """Raises `UnsupportedLanguageError` for a language with no rules yet."""

    try:
        resolved = SourceLanguage(language)
    except ValueError:
        msg = f"unknown language {language!r}"
        raise UnsupportedLanguageError(msg) from None
    rules = _RULES.get(resolved)
    if rules is None:
        msg = f"no static-check rules for {resolved.value!r} yet"
        raise UnsupportedLanguageError(msg)
    return rules
