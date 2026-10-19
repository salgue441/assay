"""
Typed settings, loaded from the YAML config with environment overrides.

Sections are typed as their stage is built; sections for stages that do not
exist yet are kept as plain mappings so the config file can describe the
whole study from the start.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from assay.corpus.discovery import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_BYTES
from assay.corpus.manifest import Manifest, ManifestError, load_manifest
from assay.corpus.units import UNIT_KINDS, UnitPolicy
from assay.extract import SourceLanguage, SymbolKind

ENV_PREFIX = "ASSAY_"
_NESTING = "__"


class SettingsError(ValueError):
    """Raised when the configuration cannot be read or is invalid."""


def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    base_dir = (info.context or {}).get("base_dir")
    return value if value.is_absolute() or base_dir is None else base_dir / value


ResolvedPath = Annotated[Path, AfterValidator(_resolve_path)]
"""A path; relative values resolve against the `base_dir` given at load time."""


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunSettings(_Section):
    seed: int
    budget_usd: PositiveFloat | None = None
    """Hard cap on LLM spend per run; None means no cap."""


class StorageSettings(_Section):
    root: ResolvedPath
    duckdb_path: ResolvedPath


class UnitSettings(_Section):
    kinds: frozenset[SymbolKind] = UNIT_KINDS
    min_lines: int = Field(default=3, ge=1)
    include_private: bool = False

    @model_validator(mode="after")
    def _check_policy(self) -> Self:
        # UnitPolicy owns the rules; building it here reports a bad config at load time.
        self.policy()
        return self

    def policy(self) -> UnitPolicy:
        return UnitPolicy(
            kinds=self.kinds, min_lines=self.min_lines, include_private=self.include_private
        )


class CorpusSettings(_Section):
    manifest: ResolvedPath
    cache_dir: ResolvedPath
    allowed_licenses: frozenset[str] = Field(min_length=1)
    languages: frozenset[SourceLanguage] = Field(min_length=1)
    units: UnitSettings = UnitSettings()
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS
    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, ge=1)

    def load_manifest(self) -> Manifest:
        """
        Load the manifest and check it against these settings: every license
        must be allowed and every language enabled.
        """

        manifest = load_manifest(self.manifest)
        manifest.check_licenses(self.allowed_licenses)

        disabled = sorted(
            f"{r.name} ({r.language.value})"
            for r in manifest.repos
            if r.language not in self.languages
        )
        if disabled:
            msg = f"{self.manifest}: languages not enabled in config: {', '.join(disabled)}"
            raise ManifestError(msg)
        return manifest


class Settings(_Section):
    run: RunSettings
    storage: StorageSettings
    corpus: CorpusSettings
    generation: dict[str, Any] = {}
    judge: dict[str, Any] = {}
    scoring: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    drift: dict[str, Any] = {}


def load_settings(
    path: Path,
    *,
    base_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """
    Load settings from a YAML file, then apply `ASSAY_<SECTION>__<KEY>`
    overrides from `environ` (default: the process environment).

    Override values are parsed as YAML, so `3`, `true`, `null` and `[a, b]`
    keep their types; empty values are ignored. Relative paths resolve
    against `base_dir` (default: the current directory). Any problem raises
    `SettingsError` naming the file.
    """

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data = _with_overrides(raw, os.environ if environ is None else environ)
        return Settings.model_validate(data, context={"base_dir": base_dir or Path.cwd()})
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        msg = f"{path}: {exc}"
        raise SettingsError(msg) from exc


def _with_overrides(raw: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(dict(raw))
    for name, value in sorted(environ.items()):
        if not name.startswith(ENV_PREFIX) or _NESTING not in name or not value.strip():
            continue

        *parents, leaf = name.removeprefix(ENV_PREFIX).lower().split(_NESTING)
        target = data
        for key in parents:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                msg = f"{name}: '{key}' is a value, not a section"
                raise SettingsError(msg)
            target = child
        target[leaf] = yaml.safe_load(value)
    return data
