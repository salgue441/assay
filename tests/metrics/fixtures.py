"""A hand-built module and the metric contexts for its units."""

from pathlib import PurePosixPath

from assay.corpus import SourceFile, Unit, UnitPolicy, select_units
from assay.extract import PythonExtractor, Symbol
from assay.metrics import MetricContext, PythonRules

SOURCE = '''"""Text helpers."""

import os


class Wrapper:
    """Wraps text."""

    def __init__(self, width: int = 78) -> None:
        self.width = width

    def wrap(self, text: str) -> list[str]:
        return [text]

    def _helper(self) -> None:
        pass


def wrap_text(text: str, width: int = 78, *, preserve: bool = False) -> str:
    """Wrap text."""
    return text


def log(message: str) -> None:
    print(message)


def ping():
    return os.getpid()
'''


def _build() -> tuple[dict[str, Unit], tuple[Symbol, ...]]:
    file = SourceFile(path=PurePosixPath("textkit/core.py"), module="textkit.core", text=SOURCE)
    inventory = PythonExtractor().extract(file.text, module=file.module, path=str(file.path))
    units = select_units(
        inventory, file, repo="owner/textkit", commit="c" * 40, policy=UnitPolicy(min_lines=1)
    )
    return {
        u.qualified_name.removeprefix("textkit.core").lstrip(".") or "module": u for u in units
    }, inventory.symbols


UNITS, SYMBOLS = _build()


def context(name: str) -> MetricContext:
    """The context for `module`, `Wrapper`, `wrap_text`, `log` or `ping`."""

    return MetricContext.build(UNITS[name], SYMBOLS, PythonRules())
