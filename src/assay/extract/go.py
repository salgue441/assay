"""
Go symbol extraction with tree-sitter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import tree_sitter_go as tsgo
from tree_sitter import Language, Node, Parser

from assay.extract._symbols import merge_redefinitions, normalize_signature
from assay.extract._treesitter import end_line, node_text, start_line, text_without_comments
from assay.extract.models import SourceLanguage, Symbol, SymbolInventory, SymbolKind

_LANGUAGE = Language(tsgo.language())
_CLASS_TYPES = frozenset({"struct_type", "interface_type"})
_TYPE_BODIES = frozenset({"field_declaration_list", "{"})
_RECEIVER_WRAPPERS = frozenset({"pointer_type", "generic_type", "parenthesized_type"})

_DIRECTIVE = re.compile(r"^//[a-z0-9_]+:\S")
"""Tool directives such as `//go:generate` or `//nolint:errcheck` are not documentation."""


class GoExtractor:
    """
    Extract the package, functions, methods, struct and interface types
    from one Go file.

    Struct and interface types are extracted as classes, and the methods of
    an interface as its methods. A method is qualified by its receiver type
    without pointer or type arguments (`module.Server.Start` for
    `func (s *Server) Start()`), whether or not that type is declared in the
    same file. Other named types (`type ID int`) and type aliases are skipped.

    A docstring is the block of `//` comment lines directly above a
    declaration, with the comment markers removed; for a type declared in a
    `type (...)` group it is the block above the type, or above the group
    when the group declares only that type. The package docstring is the
    block above the `package` clause.

    A symbol is public when it is exported: its name starts with an
    upper-case letter. The package itself is always public.

    Go forbids redefinitions except for `init`; repeated `init` functions
    are merged into the first.

    An instance owns a tree-sitter parser and is not thread-safe; use one
    per thread.
    """

    def __init__(self) -> None:
        self._parser = Parser(_LANGUAGE)

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.GO

    def extract(self, source: str, *, module: str, path: str = "") -> SymbolInventory:
        data = source.encode("utf-8")
        root = self._parser.parse(data).root_node
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
        package = next((c for c in root.named_children if c.type == "package_clause"), None)
        yield Symbol(
            qualified_name=module,
            name=module.rsplit(".", 1)[-1],
            kind=SymbolKind.MODULE,
            signature=None,
            docstring=self._doc_comment(package) if package is not None else None,
            start_line=1,
            end_line=end_line(root),
            is_public=True,
        )

        for node in root.named_children:
            if node.type == "function_declaration":
                yield from self._function(node, scope=module, kind=SymbolKind.FUNCTION)
            elif node.type == "method_declaration":
                receiver = self._receiver_type(node)
                if receiver is not None:
                    yield from self._function(
                        node, scope=f"{module}.{receiver}", kind=SymbolKind.METHOD
                    )
            elif node.type == "type_declaration":
                yield from self._types(node, scope=module)

    def _function(self, node: Node, *, scope: str, kind: SymbolKind) -> Iterator[Symbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:  # pragma: no cover - the grammar requires a name
            return

        body = node.child_by_field_name("body")
        header_end = body.start_byte if body is not None else node.end_byte
        header = text_without_comments(
            node, self._data, start_byte=node.start_byte, end_byte=header_end
        )
        yield self._symbol(
            node_text(name_node, self._data),
            scope=scope,
            kind=kind,
            signature=normalize_signature(header),
            docstring=self._doc_comment(node),
            span=node,
        )

    def _types(self, declaration: Node, *, scope: str) -> Iterator[Symbol]:
        specs = [c for c in declaration.named_children if c.type == "type_spec"]
        for spec in specs:
            name_node = spec.child_by_field_name("name")
            type_node = spec.child_by_field_name("type")
            if name_node is None or type_node is None or type_node.type not in _CLASS_TYPES:
                continue

            body = next((c for c in type_node.children if c.type in _TYPE_BODIES), None)
            header_end = body.start_byte if body is not None else spec.end_byte
            header = text_without_comments(
                spec, self._data, start_byte=spec.start_byte, end_byte=header_end
            )
            is_alone = len(specs) == 1
            docstring = self._doc_comment(spec)
            if docstring is None and is_alone:
                docstring = self._doc_comment(declaration)

            symbol = self._symbol(
                node_text(name_node, self._data),
                scope=scope,
                kind=SymbolKind.CLASS,
                signature=normalize_signature(f"type {header}"),
                docstring=docstring,
                span=declaration if is_alone else spec,
            )
            yield symbol

            if type_node.type == "interface_type":
                yield from self._interface_methods(type_node, scope=symbol.qualified_name)

    def _interface_methods(self, interface: Node, *, scope: str) -> Iterator[Symbol]:
        for element in interface.named_children:
            name_node = element.child_by_field_name("name")
            if element.type != "method_elem" or name_node is None:
                continue

            text = text_without_comments(
                element, self._data, start_byte=element.start_byte, end_byte=element.end_byte
            )
            yield self._symbol(
                node_text(name_node, self._data),
                scope=scope,
                kind=SymbolKind.METHOD,
                signature=normalize_signature(text),
                docstring=self._doc_comment(element),
                span=element,
            )

    def _symbol(
        self,
        name: str,
        *,
        scope: str,
        kind: SymbolKind,
        signature: str,
        docstring: str | None,
        span: Node,
    ) -> Symbol:
        return Symbol(
            qualified_name=f"{scope}.{name}",
            name=name,
            kind=kind,
            signature=signature,
            docstring=docstring,
            start_line=start_line(span),
            end_line=end_line(span),
            is_public=name[:1].isupper(),
        )

    def _receiver_type(self, method: Node) -> str | None:
        """
        The receiver's type name, without pointer, parentheses or type
        arguments: `Box` for `func (b *Box[T]) Get()`.
        """

        receiver = method.child_by_field_name("receiver")
        parameter = receiver.named_children[0] if receiver and receiver.named_children else None
        node = parameter.child_by_field_name("type") if parameter is not None else None
        while node is not None and node.type in _RECEIVER_WRAPPERS:
            node = node.child_by_field_name("type") or next(iter(node.named_children), None)

        if node is None or node.type != "type_identifier":
            return None
        return node_text(node, self._data)

    def _doc_comment(self, node: Node) -> str | None:
        """
        The `//` comment lines directly above `node`, with no blank line
        between them or before `node`.
        """

        comments: list[str] = []
        below = node
        comment = node.prev_sibling
        while comment is not None and _is_doc_line(comment, below, self._data):
            comments.append(node_text(comment, self._data))
            below = comment
            comment = comment.prev_sibling

        return _parse_line_comments(reversed(comments))


def _is_doc_line(comment: Node, below: Node, data: bytes) -> bool:
    """
    Whether `comment` is a `//` line directly above `below` and alone on its
    line, so it can be part of a doc comment.
    """

    if comment.type != "comment":
        return False
    if not node_text(comment, data).startswith("//"):
        return False
    if comment.end_point.row + 1 != below.start_point.row:
        return False

    previous = comment.prev_sibling
    return previous is None or previous.end_point.row != comment.start_point.row


def _parse_line_comments(comments: Iterable[str]) -> str | None:
    """
    The text of `//` comment lines without their markers and directives.
    """

    lines: list[str] = []
    for comment in comments:
        if _DIRECTIVE.match(comment):
            continue
        text = comment[2:]
        lines.append((text[1:] if text.startswith(" ") else text).rstrip())

    return "\n".join(lines).strip("\n") or None
