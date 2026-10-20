"""
TypeScript symbol extraction with tree-sitter.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from pathlib import PurePosixPath

import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from assay.extract._symbols import merge_redefinitions, normalize_signature
from assay.extract._treesitter import end_line, node_text, start_line, text_without_comments
from assay.extract.models import SourceLanguage, Symbol, SymbolInventory, SymbolKind

_TYPESCRIPT = Language(tsts.language_typescript())
_TSX = Language(tsts.language_tsx())
_TSX_SUFFIX = ".tsx"

_FUNCTIONS = frozenset({"function_declaration", "generator_function_declaration"})
_FUNCTION_OVERLOADS = frozenset({"function_signature"})
_CLASSES = frozenset({"class_declaration", "abstract_class_declaration", "interface_declaration"})
_VARIABLES = frozenset({"lexical_declaration", "variable_declaration"})
_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression", "generator_function"})
_METHODS = frozenset({"method_definition", "method_signature", "abstract_method_signature"})
_FIELDS = frozenset({"public_field_definition"})
_MEMBER_NAMES = frozenset({"property_identifier", "private_property_identifier", "identifier"})

_JSDOC_LINE_PREFIX = re.compile(r"^\s*\* ?")


class TypeScriptExtractor:
    """
    Extract modules, classes, interfaces, functions and methods from
    TypeScript source.

    Top-level declarations are included, exported or not: function
    declarations, functions bound to a variable (`const f = () => ...`,
    `const f = function () {...}`), classes (abstract ones too) and
    interfaces, which are the documentable contracts of TypeScript code and
    are extracted as classes with their method signatures as methods. Class
    members are methods, constructors, accessors and fields initialised with
    a function. Type aliases, enums and namespaces are skipped: they have no
    signature to check documentation against.

    A docstring is the JSDoc block (`/** ... */`) immediately before a
    declaration, its `export` statement or its decorators, with the comment
    markers and `*` line prefixes removed. A module's docstring is a JSDoc
    block that opens the file and documents no declaration.

    Exported top-level declarations (`export ...`, `export { name }`,
    `export default name`) and class members not marked `private` or named
    `#name` are public; a leading underscore is never public.

    Overloads, accessor pairs and declaration merging define one name more
    than once; the first definition is kept and a later docstring fills in
    only if the first has none.

    `.tsx` files are parsed with the TSX grammar, chosen by the `path`
    suffix: the two grammars disagree on `<T>expr` casts and JSX, so neither
    parses both dialects. An instance owns its parsers and is not
    thread-safe; use one per thread.
    """

    def __init__(self) -> None:
        self._typescript = Parser(_TYPESCRIPT)
        self._tsx = Parser(_TSX)

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.TYPESCRIPT

    def extract(self, source: str, *, module: str, path: str = "") -> SymbolInventory:
        data = source.encode("utf-8")
        parser = self._tsx if PurePosixPath(path).suffix == _TSX_SUFFIX else self._typescript
        root = parser.parse(data).root_node
        symbols = _SymbolCollector(data).collect(root, module=module)

        return SymbolInventory(
            module=module,
            path=path,
            language=self.language,
            has_syntax_errors=root.has_error,
            symbols=merge_redefinitions(symbols),
        )


class _SymbolCollector:
    """
    Walks one parsed file and turns its declarations into symbols.
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
            docstring=self._module_docstring(root),
            start_line=1,
            end_line=end_line(root),
            is_public=not name.startswith("_"),
        )

        exported = self._exported_names(root)
        for statement in root.named_children:
            declaration, is_exported = _unwrap_export(statement)
            if declaration is not None:
                yield from self._declaration(
                    declaration, statement, scope=module, exported=is_exported, names=exported
                )

    def _declaration(
        self, node: Node, statement: Node, *, scope: str, exported: bool, names: frozenset[str]
    ) -> Iterator[Symbol]:
        if node.type in _VARIABLES:
            yield from self._bound_functions(node, statement, scope=scope, names=names)
            return

        if node.type in _FUNCTIONS or node.type in _FUNCTION_OVERLOADS:
            kind = SymbolKind.FUNCTION
        elif node.type in _CLASSES:
            kind = SymbolKind.CLASS
        else:
            return

        name_node = node.child_by_field_name("name")
        if name_node is None or name_node.is_missing:  # pragma: no cover - error recovery only
            return

        name = node_text(name_node, self._data)
        symbol = self._symbol(
            node,
            name=name,
            scope=scope,
            kind=kind,
            signature=self._signature(node, end_byte=_body_start(node)),
            docstring=self._jsdoc_before(statement),
            is_public=(exported or name in names) and not name.startswith("_"),
        )
        yield symbol

        body = node.child_by_field_name("body")
        if kind is SymbolKind.CLASS and body is not None:
            yield from self._members(body, scope=symbol.qualified_name)

    def _bound_functions(
        self, declaration: Node, statement: Node, *, scope: str, names: frozenset[str]
    ) -> Iterator[Symbol]:
        """
        Functions bound to variables: `const name = (...) => ...`.
        """

        declarators = [c for c in declaration.named_children if c.type == "variable_declarator"]
        keyword = node_text(declaration.children[0], self._data)
        exported = statement.type == "export_statement"
        for declarator in declarators:
            name_node = declarator.child_by_field_name("name")
            value = declarator.child_by_field_name("value")
            if name_node is None or name_node.type != "identifier":
                continue
            if value is None or value.type not in _FUNCTION_VALUES:
                continue

            name = node_text(name_node, self._data)
            span = declaration if len(declarators) == 1 else declarator
            header = self._header(declarator, end_byte=_body_start(value))
            yield Symbol(
                qualified_name=f"{scope}.{name}",
                name=name,
                kind=SymbolKind.FUNCTION,
                signature=_function_value_signature(f"{keyword} {header}"),
                docstring=self._jsdoc_before(statement),
                start_line=start_line(span),
                end_line=end_line(span),
                is_public=(exported or name in names) and not name.startswith("_"),
            )

    def _members(self, body: Node, *, scope: str) -> Iterator[Symbol]:
        for member in body.named_children:
            name_node = member.child_by_field_name("name")
            if name_node is None or name_node.type not in _MEMBER_NAMES:
                continue
            if name_node.is_missing:  # inserted by error recovery; it has no text
                continue

            signature = self._member_signature(member)
            if signature is None:
                continue

            name = node_text(name_node, self._data)
            yield self._symbol(
                member,
                name=name,
                scope=scope,
                kind=SymbolKind.METHOD,
                signature=signature,
                docstring=self._jsdoc_before(member),
                is_public=_is_public_member(member, name, self._data),
            )

    def _member_signature(self, member: Node) -> str | None:
        """
        The signature of a method, or of a field initialised with a function;
        None for any other member.
        """

        if member.type in _METHODS:
            return self._signature(member, end_byte=_body_start(member))

        value = member.child_by_field_name("value")
        if member.type in _FIELDS and value is not None and value.type in _FUNCTION_VALUES:
            return _function_value_signature(self._header(member, end_byte=_body_start(value)))
        return None

    def _symbol(
        self,
        node: Node,
        *,
        name: str,
        scope: str,
        kind: SymbolKind,
        signature: str,
        docstring: str | None,
        is_public: bool,
    ) -> Symbol:
        return Symbol(
            qualified_name=f"{scope}.{name}",
            name=name,
            kind=kind,
            signature=signature,
            docstring=docstring,
            start_line=start_line(_first_undecorated(node)),
            end_line=end_line(node),
            is_public=is_public,
        )

    def _signature(self, node: Node, *, end_byte: int) -> str:
        return normalize_signature(self._header(node, end_byte=end_byte)).removesuffix(";")

    def _header(self, node: Node, *, end_byte: int) -> str:
        """
        Source of a declaration from its first keyword (after any decorators)
        up to its body, comments removed.
        """

        start = _first_undecorated(node).start_byte
        return text_without_comments(node, self._data, start_byte=start, end_byte=end_byte)

    def _jsdoc_before(self, node: Node) -> str | None:
        previous = node.prev_sibling
        while previous is not None and previous.type == "decorator":
            previous = previous.prev_sibling
        if previous is None or previous.type != "comment":
            return None
        return _parse_jsdoc(node_text(previous, self._data))

    def _module_docstring(self, root: Node) -> str | None:
        """
        A JSDoc block that opens the file, unless it documents the
        declaration right after it.
        """

        first = next(iter(root.children), None)
        if first is None or first.type != "comment":
            return None

        following = first.next_sibling
        if following is not None and _documents(following):
            return None
        return _parse_jsdoc(node_text(first, self._data))

    def _exported_names(self, root: Node) -> frozenset[str]:
        """
        Local names exported by `export { name }` or `export default name`.
        """

        names: set[str] = set()
        for statement in root.named_children:
            if statement.type != "export_statement":
                continue
            if statement.child_by_field_name("source") is not None:
                continue  # a re-export from another module

            value = statement.child_by_field_name("value")
            if value is not None and value.type == "identifier":
                names.add(node_text(value, self._data))

            for clause in statement.named_children:
                if clause.type != "export_clause":
                    continue
                for specifier in clause.named_children:
                    name = specifier.child_by_field_name("name")
                    if name is not None:
                        names.add(node_text(name, self._data))

        return frozenset(names)


def _unwrap_export(statement: Node) -> tuple[Node | None, bool]:
    """
    The declaration a top-level statement makes, and whether it is exported.
    """

    if statement.type == "export_statement":
        return statement.child_by_field_name("declaration"), True
    return statement, False


def _documents(statement: Node) -> bool:
    """
    Whether a JSDoc block right before `statement` belongs to a declaration.
    """

    declaration, _ = _unwrap_export(statement)
    if declaration is None:
        return False
    return declaration.type in _FUNCTIONS | _FUNCTION_OVERLOADS | _CLASSES | _VARIABLES


def _first_undecorated(node: Node) -> Node:
    """
    The first child of a declaration that is not a decorator, so headers
    and line ranges start at the declaration itself as in Python.
    """

    return next(
        (child for child in node.children if child.type not in ("decorator", "comment")), node
    )


def _body_start(node: Node) -> int:
    body = node.child_by_field_name("body")
    return body.start_byte if body is not None else node.end_byte


def _function_value_signature(header: str) -> str:
    """
    A function bound to a name, without the arrow that precedes its body.
    """

    return normalize_signature(header).removesuffix("=>").rstrip()


def _is_public_member(member: Node, name: str, data: bytes) -> bool:
    if name.startswith(("#", "_")):
        return False
    return not any(
        child.type == "accessibility_modifier" and node_text(child, data) == "private"
        for child in member.children
    )


def _parse_jsdoc(comment: str) -> str | None:
    """
    The text of a `/** ... */` block with its markers and `*` line prefixes
    removed, or None for any other comment.
    """

    if not comment.startswith("/**") or comment == "/**/" or not comment.endswith("*/"):
        return None

    body = comment[3:-2]
    lines = [_JSDOC_LINE_PREFIX.sub("", line).rstrip() for line in body.splitlines()]
    return inspect.cleandoc("\n".join(lines)) or None
