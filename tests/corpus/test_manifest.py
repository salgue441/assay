from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from assay.corpus import DisallowedLicenseError, Manifest, ManifestError, RepoSpec, load_manifest
from assay.extract import SourceLanguage

SHA_A = "a" * 40
SHA_B = "b" * 40


def make_spec(url: str = "https://github.com/owner/project", **overrides: object) -> RepoSpec:
    fields: dict[str, object] = {
        "url": url,
        "license": "MIT",
        "language": "python",
        "created_at": "2025-01-01",
        "epochs": [SHA_A],
    }
    return RepoSpec.model_validate(fields | overrides)


def test_load_parses_a_manifest(tmp_path: Path) -> None:
    path = tmp_path / "repos.yaml"
    path.write_text(
        f"""
repos:
  - url: https://github.com/owner/project.git
    license: Apache-2.0
    language: python
    created_at: 2024-03-01
    epochs:
      - {SHA_A}
      - {SHA_B}
""",
        encoding="utf-8",
    )

    (repo,) = load_manifest(path).repos

    assert repo.name == "owner/project"
    assert repo.slug == "owner__project"
    assert repo.language is SourceLanguage.PYTHON
    assert repo.created_at == date(2024, 3, 1)
    assert repo.epochs == (SHA_A, SHA_B)


@pytest.mark.parametrize("content", ["", "repos: []\n"])
def test_load_accepts_an_empty_manifest(tmp_path: Path, content: str) -> None:
    path = tmp_path / "repos.yaml"
    path.write_text(content, encoding="utf-8")

    assert load_manifest(path).repos == ()


def test_the_committed_manifest_template_is_valid() -> None:
    load_manifest(Path(__file__).parents[2] / "config" / "repos.yaml")


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("repos: [\n", "while parsing"),
        ("repos:\n  - url: https://github.com/o/r\n", "Field required"),
    ],
)
def test_load_names_the_file_on_error(tmp_path: Path, content: str, problem: str) -> None:
    path = tmp_path / "repos.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestError, match=problem) as excinfo:
        load_manifest(path)
    assert str(path) in str(excinfo.value)


def test_load_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "epochs",
    [[], ["abc123"], ["A" * 40], [SHA_A, SHA_A]],
    ids=["none", "abbreviated", "uppercase", "repeated"],
)
def test_rejects_invalid_epochs(epochs: list[str]) -> None:
    with pytest.raises(ValidationError):
        make_spec(epochs=epochs)


@pytest.mark.parametrize("url", ["github.com/o/r", "ssh://git@github.com/o/r", "-upload-pack=x"])
def test_rejects_urls_that_are_not_https_or_file(url: str) -> None:
    with pytest.raises(ValidationError):
        make_spec(url)


def test_name_strips_trailing_slash_and_git_suffix() -> None:
    assert make_spec("https://github.com/owner/project.git/").name == "owner/project"


def test_rejects_the_same_repository_twice() -> None:
    with pytest.raises(ValidationError, match="more than once: owner/project"):
        Manifest(repos=(make_spec(), make_spec("https://github.com/owner/project.git")))


def test_check_licenses_names_every_offender() -> None:
    manifest = Manifest(
        repos=(
            make_spec("https://github.com/a/ok"),
            make_spec("https://github.com/b/gpl", license="GPL-3.0"),
            make_spec("https://github.com/c/none", license="Proprietary"),
        )
    )

    with pytest.raises(DisallowedLicenseError, match=r"b/gpl \(GPL-3.0\), c/none \(Proprietary\)"):
        manifest.check_licenses({"MIT", "Apache-2.0"})


def test_check_licenses_passes_when_all_are_allowed() -> None:
    Manifest(repos=(make_spec(),)).check_licenses({"MIT"})
