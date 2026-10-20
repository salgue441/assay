"""
Python symbol extraction with tree-sitter.
"""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Iterable, Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from assay.extract._treesitter import descendants_before, end_line, node_text, start_line
from assay.extract.models import SourceLanguage, Symbol, SymbolInventory, SymbolKind

_LANGUAGE = Language(tspython.language())
_DEFINITIONS = frozenset({"function_definition", "class_definition"})
_STRING_LITERALS = frozenset({"string", "concatenated_string"})

# Applied in order to a header with comments removed, before whitespace is collapsed.
_SIGNATURE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r",\s*\n\s*([)\]])"), r"\1"),  # trailing comma before a closing bracket
    (re.compile(r"([(\[])\s*\n\s*"), r"\1"),  # line break after an opening bracket
    (re.compile(r"\s*\n\s*([)\]])"), r"\1"),  # line break before a closing bracket
)


class PythonExtractor:
    """
    Extract modules, classes, functions and methods from Python source.

    Top-level functions, classes and class members (including nested classes)
    are included. Definitions inside function bodies or inside control flow
    (`if`, `try`) are implementation detail and are skipped.

    When a name is defined more than once in the same scope (a property getter
    and setter, `@overload` variants), the first definition is kept and a later
    docstring fills in only if the first has none.

    An instance owns a tree-sitter parser and is not thread-safe; use one per
    thread.
    """

    def __init__(self) -> None:
        self._parser = Parser(_LANGUAGE)

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.PYTHON

    def extract(self, source: str, *, module: str, path: str = "") -> SymbolInventory:
        data = source.encode("utf-8")
        root = self._parser.parse(data).root_node
        symbols = _SymbolCollector(data).collect(root, module=module)

        return SymbolInventory(
            module=module,
            path=path,
            language=self.language,
            has_syntax_errors=root.has_error,
            symbols=_merge_redefinitions(symbols),
        )


class _SymbolCollector:
    """
    Walks one parsed file and turns its definitions into symbols.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    def collect(self, root: Node, *, module: str) -> Iterator[Symbol]:
        name = module.rsplit(".", 1)[-1]
        yield Symbol(
            qualified_name=module,
            name=name,
            kind=SymbolKind.MODULE,
            signature=None,
            docstring=self._docstring(root),
            start_line=1,
            end_line=end_line(root),
            is_public=_is_public(name),
        )
        yield from self._definitions(root, scope=module, in_class=False)

    def _definitions(self, block: Node, *, scope: str, in_class: bool) -> Iterator[Symbol]:
        for child in block.named_children:
            node = _unwrap_decorated(child)
            if node.type not in _DEFINITIONS:
                continue

            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None or body is None:  # pragma: no cover - the grammar requires both
                continue

            name = node_text(name_node, self._data)
            qualified_name = f"{scope}.{name}"
            kind = _kind(node, in_class=in_class)

            yield Symbol(
                qualified_name=qualified_name,
                name=name,
                kind=kind,
                signature=_normalize_signature(self._header(node)),
                docstring=self._docstring(body),
                start_line=start_line(node),
                end_line=end_line(node),
                is_public=_is_public(name),
            )

            if kind is SymbolKind.CLASS:
                yield from self._definitions(body, scope=qualified_name, in_class=True)

    def _header(self, node: Node) -> str:
        """
        Source of a definition from its keyword up to its colon, comments removed.
        """

        colon = next((child for child in node.children if child.type == ":"), None)
        header_end = colon.start_byte if colon is not None else node.end_byte

        pieces: list[bytes] = []
        cursor = node.start_byte
        for descendant in descendants_before(node, header_end):
            if descendant.type == "comment":
                pieces.append(self._data[cursor : descendant.start_byte])
                cursor = descendant.end_byte
        pieces.append(self._data[cursor:header_end])

        return b"".join(pieces).decode("utf-8")

    def _docstring(self, block: Node) -> str | None:
        """
        The docstring of a module or block: its first statement, if that
        statement is a lone string literal.
        """

        first = next(iter(block.named_children), None)
        if first is None or first.type != "expression_statement" or first.named_child_count != 1:
            return None

        expression = first.named_children[0]
        if expression.type not in _STRING_LITERALS:
            return None

        return _parse_docstring(node_text(expression, self._data))


def _is_public(name: str) -> bool:
    """
    Public by convention: no leading underscore on the name itself.
    """

    return not name.startswith("_")


def _kind(node: Node, *, in_class: bool) -> SymbolKind:
    if node.type == "class_definition":
        return SymbolKind.CLASS

    return SymbolKind.METHOD if in_class else SymbolKind.FUNCTION


def _unwrap_decorated(node: Node) -> Node:
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            return inner

    return node


def _normalize_signature(header: str) -> str:
    """
    Collapse a possibly multi-line definition header onto one line.
    """

    for pattern, replacement in _SIGNATURE_RULES:
        header = pattern.sub(replacement, header)

    return " ".join(header.split())


def _parse_docstring(literal: str) -> str | None:
    """
    Evaluate a string literal and clean it the way `ast.get_docstring` does.

    f-strings, bytes and malformed literals are not docstrings.
    """

    try:
        value = ast.literal_eval(literal)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(value, str):
        return None

    return inspect.cleandoc(value) or None


def _merge_redefinitions(symbols: Iterable[Symbol]) -> tuple[Symbol, ...]:
    merged: dict[str, Symbol] = {}
    for symbol in symbols:
        first = merged.get(symbol.qualified_name)
        if first is None:
            merged[symbol.qualified_name] = symbol
        elif first.docstring is None and symbol.docstring is not None:
            merged[symbol.qualified_name] = first.model_copy(update={"docstring": symbol.docstring})

    return tuple(merged.values())
