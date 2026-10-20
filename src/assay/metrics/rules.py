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
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Protocol

import tree_sitter_go as tsgo
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from assay.extract import SourceLanguage, UnsupportedLanguageError

_RECEIVERS = frozenset({"self", "cls"})


class LanguageRules(Protocol):
    @property
    def ambient_names(self) -> frozenset[str]:
        """Names any code can use without defining them: keywords, builtins,
        standard library modules and the attributes of builtin types."""
        ...

    @property
    def constructor_name(self) -> str | None:
        """The method name of a class constructor, or None if the language
        has no constructor methods."""
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

    @property
    def constructor_name(self) -> str:
        return "__init__"

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


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# TypeScript: keywords, JavaScript and DOM globals, utility types, and the
# members of built-in objects and prototypes.
_TYPESCRIPT_AMBIENT_NAMES = _words(
    """
    abstract any as asserts async await bigint boolean break case catch class const
    constructor continue debugger declare default delete do else enum export extends
    false finally for from function get global if implements import in infer instanceof
    interface is keyof let module namespace never new null number object of override
    package private protected public readonly require return satisfies set static string
    super switch symbol this throw true try type typeof undefined unique unknown var void
    while with yield

    AbortController AbortSignal AggregateError Array ArrayBuffer ArrayLike AsyncGenerator
    AsyncIterable AsyncIterableIterator Awaited BigInt Blob Boolean Buffer ConstructorParameters
    DataView Date Document Element Error EvalError Event EventTarget Exclude Extract
    Float32Array Float64Array FormData Function Generator Headers HTMLElement Infinity
    InstanceType Int16Array Int32Array Int8Array Intl Iterable IterableIterator Iterator JSON
    Map Math NaN Node NonNullable Number Object Omit Parameters Partial Pick Promise
    PromiseLike PropertyKey Proxy RangeError Readonly ReadonlyArray ReadonlyMap ReadonlySet
    Record ReferenceError Reflect RegExp Request Required Response ReturnType Set String
    Symbol SyntaxError TextDecoder TextEncoder ThisType TypeError URIError URL URLSearchParams
    Uint16Array Uint32Array Uint8Array Uint8ClampedArray WeakMap WeakRef WeakSet Window
    clearInterval clearTimeout console decodeURI decodeURIComponent document encodeURI
    encodeURIComponent exports fetch globalThis isFinite isNaN parseFloat parseInt process
    queueMicrotask setInterval setTimeout structuredClone window

    apply bind call hasOwnProperty isPrototypeOf length propertyIsEnumerable prototype
    toLocaleString toString valueOf
    at concat copyWithin entries every fill filter find findIndex findLast findLastIndex
    flat flatMap forEach from includes indexOf isArray join keys lastIndexOf map pop push
    reduce reduceRight reverse shift slice some sort splice toReversed toSorted toSpliced
    unshift values with
    charAt charCodeAt codePointAt endsWith fromCharCode localeCompare match matchAll
    normalize padEnd padStart raw repeat replace replaceAll search split startsWith
    substring toLowerCase toUpperCase trim trimEnd trimStart
    add clear delete get has set size
    all allSettled any catch finally race reject resolve then
    assign create defineProperty freeze fromEntries getOwnPropertyNames getPrototypeOf
    is isFrozen seal
    parse stringify
    PI abs ceil floor max min pow random round sign sqrt trunc
    EPSILON MAX_SAFE_INTEGER MIN_SAFE_INTEGER isInteger isSafeInteger toFixed toPrecision
    debug error info log table trace warn
    getTime now toISOString
    asyncIterator iterator
    cause message name stack
    exec lastIndex test
    """
)

_TYPESCRIPT = Language(tsts.language_typescript())
_TYPESCRIPT_CALLABLES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function_signature",
        "function_expression",
        "generator_function",
        "arrow_function",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
    }
)
# A signature is completed into a parseable program by the first template that
# parses without errors: declarations take a body, bound functions an arrow and
# a body, class members a class around them, and abstract members no body.
_TYPESCRIPT_TEMPLATES = (
    "{} {{}}",
    "{} => {{}}",
    "class _ {{ {} {{}} }}",
    "class _ {{ {} => {{}} }}",
    "abstract class _ {{ {}; }}",
)
_TYPESCRIPT_RECEIVER = "this"
_TYPESCRIPT_NO_VALUE = "void"


class TypeScriptRules:
    """
    Parameters and return types read from the signatures the TypeScript
    extractor produces; destructured parameters have no name and are left
    out, as is the `this` parameter. A `void` return declares that nothing
    is returned, so it reads as no return type, like a missing annotation.
    """

    @property
    def ambient_names(self) -> frozenset[str]:
        return _TYPESCRIPT_AMBIENT_NAMES

    @property
    def constructor_name(self) -> str:
        return "constructor"

    def parameters(self, signature: str) -> tuple[str, ...]:
        parsed = _parse_callable(
            _TYPESCRIPT, _TYPESCRIPT_TEMPLATES, _TYPESCRIPT_CALLABLES, signature
        )
        if parsed is None:
            return ()
        callable_node, data = parsed

        single = callable_node.child_by_field_name("parameter")  # `x => ...`
        if single is not None:
            return (_text(single, data),)

        parameters = callable_node.child_by_field_name("parameters")
        names = (
            _typescript_parameter_name(p, data)
            for p in (parameters.named_children if parameters is not None else ())
        )
        return tuple(n for n in names if n is not None and n != _TYPESCRIPT_RECEIVER)

    def return_annotation(self, signature: str) -> str | None:
        parsed = _parse_callable(
            _TYPESCRIPT, _TYPESCRIPT_TEMPLATES, _TYPESCRIPT_CALLABLES, signature
        )
        if parsed is None:
            return None
        callable_node, data = parsed

        annotation = callable_node.child_by_field_name("return_type")
        if annotation is None:
            return None
        declared = _text(annotation, data).removeprefix(":").strip()
        return None if declared == _TYPESCRIPT_NO_VALUE else declared


def _typescript_parameter_name(parameter: Node, data: bytes) -> str | None:
    pattern = parameter.child_by_field_name("pattern")
    if pattern is not None and pattern.type == "rest_pattern":
        pattern = next(iter(pattern.named_children), None)
    if pattern is None or pattern.type not in ("identifier", "this"):
        return None
    return _text(pattern, data)


# Go: keywords, predeclared identifiers, standard library packages and the
# members of the packages almost every program uses.
_GO_AMBIENT_NAMES = _words(
    """
    break case chan const continue default defer else fallthrough for func go goto if
    import interface map package range return select struct switch type var

    any append bool byte cap clear close comparable complex complex128 complex64 copy
    delete error false float32 float64 imag int int16 int32 int64 int8 iota len make max
    min new nil panic print println real recover rune string true uint uint16 uint32
    uint64 uint8 uintptr Error String

    archive atomic base64 big bufio bytes cmp compress container context crypto csv
    database debug embed encoding errors exec expvar filepath flag fmt fs hash heap hex
    html http image io ioutil iter json list log maps math md5 mime net os path plugin
    rand reflect regexp ring runtime sha1 sha256 signal slices slog sort sql strconv
    strings sync syscall template testing text time tls unicode unsafe url utf16 utf8 xml

    Errorf Fprint Fprintf Fprintln Print Printf Println Sprint Sprintf Sprintln Sscanf
    Stringer
    As Is Join New Unwrap
    Background CancelFunc Canceled Context Deadline DeadlineExceeded Done Err TODO Value
    WithCancel WithDeadline WithTimeout WithValue
    Builder Contains EqualFold Fields HasPrefix HasSuffix Index NewReader Repeat Replace
    ReplaceAll Split ToLower ToUpper Trim TrimPrefix TrimSpace TrimSuffix
    After Duration Format Hour Microsecond Millisecond Minute Nanosecond NewTicker NewTimer
    Now Parse RFC3339 Second Since Sleep Tick Ticker Time Timer Unix Until
    Close Closer Copy Discard EOF Read ReadAll ReadCloser ReadWriter Reader Write
    WriteCloser Writer
    Add Do Lock Map Mutex Once Pool RLock RUnlock RWMutex Unlock Wait WaitGroup
    Args Create Exit File Getenv Open ReadFile Stderr Stdin Stdout WriteFile
    Client Get Handler HandlerFunc ListenAndServe Post Request ResponseWriter Server StatusOK
    Atoi FormatInt Itoa ParseFloat ParseInt Quote
    Ints Slice Sort Strings
    Keys Values
    Marshal NewDecoder NewEncoder Unmarshal
    Fatal Fatalf
    Buffer
    """
)

_GO = Language(tsgo.language())
_GO_CALLABLES = frozenset({"function_declaration", "method_declaration", "method_elem"})
# Function and method headers take a body; interface methods an interface.
_GO_TEMPLATES = ("package p\n{} {{}}\n", "package p\ntype _ interface {{ {} }}\n")
_GO_PARAMETERS = frozenset({"parameter_declaration", "variadic_parameter_declaration"})


class GoRules:
    """
    Parameters and results read from the signatures the Go extractor
    produces. The receiver is not a parameter, and Go has no constructor
    methods (`NewServer` is an ordinary function).
    """

    @property
    def ambient_names(self) -> frozenset[str]:
        return _GO_AMBIENT_NAMES

    @property
    def constructor_name(self) -> None:
        return None

    def parameters(self, signature: str) -> tuple[str, ...]:
        parsed = _parse_callable(_GO, _GO_TEMPLATES, _GO_CALLABLES, signature)
        if parsed is None:
            return ()
        callable_node, data = parsed

        parameters = callable_node.child_by_field_name("parameters")
        declarations = parameters.named_children if parameters is not None else ()
        return tuple(
            _text(name, data)
            for declaration in declarations
            if declaration.type in _GO_PARAMETERS
            for name in declaration.children_by_field_name("name")
        )

    def return_annotation(self, signature: str) -> str | None:
        parsed = _parse_callable(_GO, _GO_TEMPLATES, _GO_CALLABLES, signature)
        if parsed is None:
            return None
        callable_node, data = parsed

        result = callable_node.child_by_field_name("result")
        return _text(result, data) if result is not None else None


def _parse_callable(
    language: Language, templates: tuple[str, ...], callables: frozenset[str], signature: str
) -> tuple[Node, bytes] | None:
    """
    The first callable in the first completion of a signature that parses
    and declares one, with the completed source; None if there is none.

    A parser per call keeps the shared rules thread-safe; signatures are
    short, so this is cheap.
    """

    parser = Parser(language)
    for template in templates:
        data = template.format(signature).encode("utf-8")
        root = parser.parse(data).root_node
        if root.has_error:
            continue
        found = next((node for node in _preorder(root) if node.type in callables), None)
        if found is not None:
            return found, data
    return None


def _preorder(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


def _text(node: Node, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8")


_RULES: Mapping[SourceLanguage, LanguageRules] = MappingProxyType(
    {
        SourceLanguage.PYTHON: PythonRules(),
        SourceLanguage.TYPESCRIPT: TypeScriptRules(),
        SourceLanguage.GO: GoRules(),
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
