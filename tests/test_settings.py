from datetime import date
from pathlib import Path

import pytest
import yaml

from assay.corpus import ManifestError, UnitPolicy
from assay.extract import SourceLanguage, SymbolKind
from assay.settings import Settings, SettingsError, load_settings

PROJECT_ROOT = Path(__file__).parents[1]
SHA = "a" * 40

MINIMAL = {
    "run": {"seed": 1},
    "storage": {"root": "data", "duckdb_path": "data/assay.duckdb"},
    "corpus": {
        "manifest": "repos.yaml",
        "cache_dir": "cache",
        "allowed_licenses": ["MIT"],
        "languages": ["python"],
    },
    "generation": {
        "prompt_version": "v1",
        "prompt_dir": str(PROJECT_ROOT / "config" / "prompts" / "generation"),
        "subjects": [{"name": "writer", "provider": "ollama", "model_id": "phi4-mini"}],
    },
}


def write_config(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def load(tmp_path: Path, data: object = MINIMAL, **environ: str) -> Settings:
    return load_settings(write_config(tmp_path, data), base_dir=tmp_path, environ=environ)


def test_the_committed_default_config_is_valid() -> None:
    settings = load_settings(
        PROJECT_ROOT / "config" / "default.yaml", base_dir=PROJECT_ROOT, environ={}
    )

    assert settings.storage.root == PROJECT_ROOT / "data"
    assert settings.corpus.units.policy() == UnitPolicy()
    assert settings.corpus.load_manifest().repos


def test_relative_paths_resolve_against_base_dir(tmp_path: Path) -> None:
    settings = load(tmp_path)

    assert settings.storage.root == tmp_path / "data"
    assert settings.corpus.manifest == tmp_path / "repos.yaml"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    absolute = (tmp_path / "elsewhere").resolve()
    data = MINIMAL | {"storage": {"root": str(absolute), "duckdb_path": "x.duckdb"}}

    assert load(tmp_path, data).storage.root == absolute


def test_defaults_fill_optional_sections(tmp_path: Path) -> None:
    settings = load(tmp_path)

    assert settings.run.budget_usd is None
    assert settings.corpus.units.policy() == UnitPolicy()
    assert settings.judge == {}


def test_environment_overrides_nested_values_with_yaml_types(tmp_path: Path) -> None:
    settings = load(
        tmp_path,
        ASSAY_RUN__BUDGET_USD="12.5",
        ASSAY_STORAGE__ROOT="other",
        ASSAY_CORPUS__UNITS__MIN_LINES="5",
        ASSAY_CORPUS__UNITS__KINDS="[class]",
        ASSAY_CORPUS__LANGUAGES="[python, go]",
    )

    assert settings.run.budget_usd == 12.5
    assert settings.storage.root == tmp_path / "other"
    assert settings.corpus.units.policy() == UnitPolicy(
        kinds=frozenset({SymbolKind.CLASS}), min_lines=5
    )
    assert settings.corpus.languages == {SourceLanguage.PYTHON, SourceLanguage.GO}


def test_environment_ignores_unrelated_unnested_and_empty_variables(tmp_path: Path) -> None:
    settings = load(
        tmp_path, HOME="/home/x", ASSAY_CONFIG_FILE="other.yaml", ASSAY_STORAGE__ROOT=""
    )

    assert settings.storage.root == tmp_path / "data"


def test_environment_override_of_an_unknown_key_fails(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="min_line"):
        load(tmp_path, ASSAY_CORPUS__UNITS__MIN_LINE="5")


def test_environment_cannot_nest_under_a_value(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="'seed' is a value, not a section"):
        load(tmp_path, ASSAY_RUN__SEED__X="1")


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"run": {"seed": 1, "budget_usd": 0}}, "greater than 0"),
        ({"run": {"seed": 1, "colour": "red"}}, "Extra inputs are not permitted"),
        ({"corpus": MINIMAL["corpus"] | {"languages": []}}, "at least 1 item"),
        ({"corpus": MINIMAL["corpus"] | {"units": {"kinds": ["method"]}}}, "unit kinds"),
        ({"surprise": {}}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_settings_name_the_file(
    tmp_path: Path, change: dict[str, object], problem: str
) -> None:
    with pytest.raises(SettingsError, match=problem) as excinfo:
        load(tmp_path, MINIMAL | change)
    assert "config.yaml" in str(excinfo.value)


def generation(**changes: object) -> dict[str, object]:
    return {"generation": MINIMAL["generation"] | changes}


def test_generation_settings_load_the_prompt(tmp_path: Path) -> None:
    generation_settings = load(tmp_path).generation

    assert generation_settings.prompt().version == "v1"
    assert generation_settings.max_output_tokens == 1024
    assert generation_settings.sample_per_partition is None
    assert generation_settings.subjects[0].temperature == 0.0


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (generation(prompt_version="v404"), "v404.yaml"),
        (generation(subjects=[]), "at least 1 item"),
        (
            generation(subjects=[{"name": "w", "provider": "ollama", "model_id": "m"}] * 2),
            "subject names must be unique",
        ),
        (generation(subjects=[{"name": "w", "provider": "cohere", "model_id": "m"}]), "provider"),
        (generation(sample_per_partition=0), "greater than or equal to 1"),
    ],
)
def test_invalid_generation_settings_fail_at_load(
    tmp_path: Path, change: dict[str, object], problem: str
) -> None:
    with pytest.raises(SettingsError, match=problem):
        load(tmp_path, MINIMAL | change)


def test_unreadable_files_raise_settings_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError):
        load_settings(tmp_path / "missing.yaml")

    broken = tmp_path / "broken.yaml"
    broken.write_text("run: [\n", encoding="utf-8")
    with pytest.raises(SettingsError):
        load_settings(broken)


def write_manifest(tmp_path: Path, *, license: str, language: str) -> None:
    repo = {
        "url": "https://github.com/owner/project",
        "license": license,
        "language": language,
        "created_at": date(2025, 1, 1),
        "epochs": [SHA],
    }
    write_config(tmp_path, {"repos": [repo]}).rename(tmp_path / "repos.yaml")


def test_load_manifest_accepts_allowed_licenses_and_languages(tmp_path: Path) -> None:
    write_manifest(tmp_path, license="MIT", language="python")

    (repo,) = load(tmp_path).corpus.load_manifest().repos

    assert repo.name == "owner/project"


def test_load_manifest_rejects_disallowed_licenses(tmp_path: Path) -> None:
    write_manifest(tmp_path, license="GPL-3.0", language="python")

    with pytest.raises(ValueError, match=r"owner/project \(GPL-3.0\)"):
        load(tmp_path).corpus.load_manifest()


def test_load_manifest_rejects_languages_not_enabled(tmp_path: Path) -> None:
    write_manifest(tmp_path, license="MIT", language="go")

    with pytest.raises(ManifestError, match=r"not enabled in config: owner/project \(go\)"):
        load(tmp_path).corpus.load_manifest()
