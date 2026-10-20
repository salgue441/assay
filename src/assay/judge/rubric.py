"""
Versioned judging rubrics and the reply every judge must give.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from string import Template
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assay.corpus import Unit

MIN_SCORE = 1
MAX_SCORE = 5


class RubricError(ValueError):
    """Raised when a rubric file is missing, malformed or cannot be rendered."""


@dataclass(frozen=True)
class Criterion:
    name: str
    question: str
    levels: Mapping[int, str]
    """A description per score, from MIN_SCORE to MAX_SCORE."""

    def __post_init__(self) -> None:
        if sorted(self.levels) != list(range(MIN_SCORE, MAX_SCORE + 1)):
            msg = f"criterion {self.name!r} must describe every level {MIN_SCORE}-{MAX_SCORE}"
            raise RubricError(msg)


@dataclass(frozen=True)
class Rubric:
    """
    Criteria with anchored levels and the prompt that grades one criterion,
    loaded from `<directory>/<version>.yaml`. `hash` covers the whole file
    content as parsed, so any edit gives judgments a different id.
    """

    version: str
    system: str
    user: str
    criteria: Mapping[str, Criterion]

    @classmethod
    def load(cls, directory: Path, version: str) -> Rubric:
        path = directory / f"{version}.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            criteria = {
                name: Criterion(
                    name=name,
                    question=spec["question"],
                    levels={int(level): text for level, text in spec["levels"].items()},
                )
                for name, spec in data["criteria"].items()
            }
            return cls(version=version, system=data["system"], user=data["user"], criteria=criteria)
        except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError, ValueError) as exc:
            msg = f"{path}: not a rubric with system, user and criteria ({exc})"
            raise RubricError(msg) from exc

    @cached_property
    def hash(self) -> str:
        canonical = json.dumps(
            {
                "system": self.system,
                "user": self.user,
                "criteria": {
                    name: {
                        "question": c.question,
                        "levels": {str(k): v for k, v in c.levels.items()},
                    }
                    for name, c in self.criteria.items()
                },
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def render(
        self, criterion: str, *, unit: Unit, document: str, language: str
    ) -> tuple[str, str]:
        """The (system, user) messages grading `document` on one criterion."""

        spec = self.criteria.get(criterion)
        if spec is None:
            msg = f"rubric {self.version} has no criterion {criterion!r}"
            raise RubricError(msg)

        values = {
            "kind": unit.kind.value,
            "name": unit.qualified_name,
            "language": language,
            "source": unit.source.rstrip("\n"),
            "document": document.strip(),
            "criterion": criterion,
            "question": spec.question,
            "levels": "\n".join(f"{score}: {text}" for score, text in sorted(spec.levels.items())),
        }
        try:
            return Template(self.system).substitute(values), Template(self.user).substitute(values)
        except (KeyError, ValueError) as exc:
            msg = f"rubric {self.version}: cannot render ({exc!r})"
            raise RubricError(msg) from exc


class JudgeReply(BaseModel):
    """
    What a judge returns for one criterion. Field order is the order the
    reply is written in: problems first, then reasoning, then the score.
    """

    model_config = ConfigDict(extra="forbid")

    issues: list[str]
    rationale: str
    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE)

    @classmethod
    def parse(cls, text: str) -> JudgeReply:
        """Raise `ValueError` for anything but a valid reply."""

        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            msg = f"invalid judge reply: {exc.errors()[0]['msg']}"
            raise ValueError(msg) from exc


REPLY_SCHEMA: Mapping[str, Any] = JudgeReply.model_json_schema()
"""The JSON schema judges' replies are constrained to."""
