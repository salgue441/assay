"""
Typed settings, loaded from the YAML config with environment overrides.

Sections are typed as their stage is built; sections for stages that do not
exist yet are kept as plain mappings so the config file can describe the
whole study from the start.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from assay.corpus.discovery import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_BYTES
from assay.corpus.manifest import Manifest, ManifestError, load_manifest
from assay.corpus.units import UNIT_KINDS, UnitPolicy
from assay.extract import SourceLanguage, SymbolKind
from assay.generate import PromptError, PromptTemplate
from assay.judge import Rubric, RubricError
from assay.llm import Provider
from assay.rating.ratings import RATER_ID_PATTERN

ENV_PREFIX = "ASSAY_"
_NESTING = "__"
_WEIGHT_TOLERANCE = 1e-9


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


class ModelSettings(_Section):
    """A model that writes or grades documentation."""

    name: str = Field(min_length=1)
    """Short label stored with each row it produces, such as `qwen-coder-3b`."""
    provider: Provider
    model_id: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)


class GenerationSettings(_Section):
    prompt_version: str = Field(min_length=1)
    prompt_dir: ResolvedPath
    max_output_tokens: int = Field(default=1024, ge=1)
    sample_per_partition: int | None = Field(default=None, ge=1)
    """Units documented per repository epoch; None documents every unit."""
    subjects: tuple[ModelSettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_unique_names("subject", self.subjects)
        try:
            self.prompt()
        except PromptError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def prompt(self) -> PromptTemplate:
        return PromptTemplate.load(self.prompt_dir, self.prompt_version)


class VarianceStudySettings(_Section):
    temperatures: tuple[float, ...] = Field(min_length=1)
    subset_size: int = Field(ge=1)


class JudgeSettings(_Section):
    rubric_version: str = Field(min_length=1)
    rubric_dir: ResolvedPath
    runs_per_criterion: int = Field(ge=1)
    """Repeated grades per document and criterion, to measure judge variance."""
    criteria: tuple[str, ...] = Field(min_length=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    variance_study: VarianceStudySettings
    models: tuple[ModelSettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_unique_names("judge", self.models)
        if len(set(self.criteria)) != len(self.criteria):
            msg = f"criteria must be unique: {', '.join(self.criteria)}"
            raise ValueError(msg)
        try:
            rubric = self.rubric()
        except RubricError as exc:
            raise ValueError(str(exc)) from exc
        unknown = [c for c in self.criteria if c not in rubric.criteria]
        if unknown:
            msg = f"criteria not in rubric {rubric.version}: {', '.join(unknown)}"
            raise ValueError(msg)
        return self

    def rubric(self) -> Rubric:
        return Rubric.load(self.rubric_dir, self.rubric_version)


def _require_unique_names(role: str, models: tuple[ModelSettings, ...]) -> None:
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        msg = f"{role} names must be unique: {', '.join(names)}"
        raise ValueError(msg)


class RatingSettings(_Section):
    additional_rater_share: float = Field(default=0.2, gt=0, le=1)
    """The primary rater rates every document; each additional rater rates
    this share, the same documents for all, so rater agreement can be measured."""


class CriticalFailureSettings(_Section):
    referential_integrity_below: float = Field(ge=0, le=1)
    """A document citing a smaller share of resolvable names is rejected."""


class RoutingSettings(_Section):
    publish_at_or_above: float = Field(ge=0, le=100)
    escalate_below: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.escalate_below > self.publish_at_or_above:
            msg = "escalate_below must not exceed publish_at_or_above"
            raise ValueError(msg)
        return self


class ScoringSettings(_Section):
    weights: dict[str, float] = Field(min_length=1)
    """Prior weights per metric or judged criterion; fitted weights replace
    them after calibration on the development split."""
    judge: str = Field(min_length=1)
    """The judge model, by its configured name, whose grades supply the
    judged components."""
    critical_failures: CriticalFailureSettings
    routing: RoutingSettings

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        negative = sorted(name for name, weight in self.weights.items() if weight < 0)
        if negative:
            msg = f"weights must not be negative: {', '.join(negative)}"
            raise ValueError(msg)
        if abs(sum(self.weights.values()) - 1.0) > _WEIGHT_TOLERANCE:
            msg = f"weights must sum to 1, got {sum(self.weights.values()):.6f}"
            raise ValueError(msg)
        return self


class SplitSettings(_Section):
    development_fraction: float = Field(gt=0, lt=1)
    group_by: Literal["repo"] = "repo"
    """No group appears in both splits."""
    stratify_by: tuple[str, ...] = ()


class WeightFitSettings(_Section):
    target: Literal["overall_rating"] = "overall_rating"
    shrink_to_prior: bool = True
    """Shrink fitted weights toward the prior, with the strength chosen by
    repository-grouped cross-validation on the development split."""


class TargetSettings(_Section):
    spearman_rho: float = Field(ge=-1, le=1)
    weighted_kappa: float = Field(ge=-1, le=1)
    judge_band_stability: float = Field(ge=0, le=1)


class EvaluationSettings(_Section):
    raters: tuple[str, ...] = Field(min_length=1)
    """The human raters whose ratings calibrate the score, by pseudonym.
    Ratings by anyone else, such as an LLM rater, are left out."""
    split: SplitSettings
    weight_fit: WeightFitSettings = WeightFitSettings()
    bootstrap_resamples: int = Field(default=10_000, ge=1)
    targets: TargetSettings
    golden_set: dict[str, Any] = {}
    baseline: dict[str, Any] = {}

    @field_validator("raters")
    @classmethod
    def _check_raters(cls, raters: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [r for r in raters if not re.fullmatch(RATER_ID_PATTERN, r)]
        if invalid:
            msg = f"raters must be short lowercase pseudonyms such as r1: {', '.join(invalid)}"
            raise ValueError(msg)
        if len(set(raters)) != len(raters):
            msg = f"raters must be unique: {', '.join(raters)}"
            raise ValueError(msg)
        return raters


class Settings(_Section):
    run: RunSettings
    storage: StorageSettings
    corpus: CorpusSettings
    generation: GenerationSettings
    judge: JudgeSettings
    rating: RatingSettings = RatingSettings()
    scoring: ScoringSettings
    evaluation: EvaluationSettings | None = None
    """Needed only to calibrate."""
    drift: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_scoring_judge(self) -> Self:
        judges = [model.name for model in self.judge.models]
        if self.scoring.judge not in judges:
            msg = (
                f"scoring.judge {self.scoring.judge!r} is not a configured judge; "
                f"choose one of: {', '.join(judges)}"
            )
            raise ValueError(msg)
        return self


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
