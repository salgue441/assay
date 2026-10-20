"""Real units cut from a small Python module, for generation tests."""

from pathlib import PurePosixPath

from assay.corpus import SourceFile, Unit, UnitPolicy, select_units
from assay.extract import PythonExtractor

SOURCE = '''"""Shapes and their areas."""


class Square:
    """A square."""

    def area(self) -> int:
        return 4


class Circle:
    """A circle."""


def describe(shape: object) -> str:
    """Name a shape."""
    return type(shape).__name__


def scale(factor: float) -> float:
    return factor * 2
'''


def units(source: str = SOURCE, commit: str = "c" * 40) -> list[Unit]:
    file = SourceFile(path=PurePosixPath("shapes/core.py"), module="shapes.core", text=source)
    inventory = PythonExtractor().extract(file.text, module=file.module, path=str(file.path))
    return list(
        select_units(
            inventory, file, repo="owner/shapes", commit=commit, policy=UnitPolicy(min_lines=1)
        )
    )
