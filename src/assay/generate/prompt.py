"""
Versioned prompt templates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from string import Template

import yaml

from assay.corpus import Unit


class PromptError(ValueError):
    """Raised when a prompt file is missing, malformed or cannot be rendered."""


@dataclass(frozen=True)
class PromptTemplate:
    """
    A system and user message with `$placeholders`, loaded from
    `<directory>/<version>.yaml`.

    `hash` covers both messages exactly as written, so any edit, even to
    whitespace, gives documents a different id.
    """

    version: str
    system: str
    user: str

    @classmethod
    def load(cls, directory: Path, version: str) -> PromptTemplate:
        path = directory / f"{version}.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls(version=version, system=data["system"], user=data["user"])
        except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
            msg = f"{path}: not a prompt with 'system' and 'user' messages ({exc})"
            raise PromptError(msg) from exc

    @cached_property
    def hash(self) -> str:
        canonical = json.dumps({"system": self.system, "user": self.user}, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def render(self, unit: Unit, *, language: str) -> tuple[str, str]:
        """The (system, user) messages for one unit."""

        values = {
            "kind": unit.kind.value,
            "name": unit.qualified_name,
            "repo": unit.repo,
            "path": unit.path,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
            "language": language,
            "source": unit.source.rstrip("\n"),
        }
        try:
            return (
                Template(self.system).substitute(values),
                Template(self.user).substitute(values),
            )
        except (KeyError, ValueError) as exc:
            msg = f"prompt {self.version}: cannot render ({exc!r})"
            raise PromptError(msg) from exc
