import re
from pathlib import Path

import pytest

from assay.generate import PromptError, PromptTemplate

from .units import units

PROJECT_ROOT = Path(__file__).parents[2]


def write_prompt(tmp_path: Path, content: str, version: str = "v9") -> PromptTemplate:
    (tmp_path / f"{version}.yaml").write_text(content, encoding="utf-8")
    return PromptTemplate.load(tmp_path, version)


def test_the_committed_v1_prompt_renders_every_unit_kind() -> None:
    prompt = PromptTemplate.load(PROJECT_ROOT / "config" / "prompts" / "generation", "v1")

    for unit in units():
        system, user = prompt.render(unit, language="python")
        source = unit.source.rstrip("\n")
        assert system.startswith("You are a technical writer")
        assert f"## `{unit.qualified_name}`" in user
        assert f"```python\n{source}\n```" in user
        assert "$" not in user.replace(source, "")


def test_renders_unit_fields_into_placeholders(tmp_path: Path) -> None:
    template = "system: Document $language code.\nuser: $kind $name in $repo:$path L$start_line\n"
    prompt = write_prompt(tmp_path, template)
    unit = units()[1]

    system, user = prompt.render(unit, language="python")

    assert system == "Document python code."
    assert user == f"class shapes.core.Square in owner/shapes:shapes/core.py L{unit.start_line}"


def test_dollar_signs_in_source_are_left_alone(tmp_path: Path) -> None:
    prompt = write_prompt(tmp_path, "system: s\nuser: $source costs $$5\n")
    unit = units('PRICE = "$total"\n')[0]

    assert prompt.render(unit, language="python")[1] == 'PRICE = "$total" costs $5'


def test_hash_changes_with_any_edit(tmp_path: Path) -> None:
    first = write_prompt(tmp_path, "system: s\nuser: u\n", "a")
    same = write_prompt(tmp_path, "system: s\nuser: u\n", "b")
    edited = write_prompt(tmp_path, "system: s\nuser: 'u '\n", "c")

    assert first.hash == same.hash
    assert first.hash != edited.hash
    assert len(first.hash) == 64


def test_unknown_placeholders_fail_to_render(tmp_path: Path) -> None:
    prompt = write_prompt(tmp_path, "system: s\nuser: $nonsense\n")

    with pytest.raises(PromptError, match="cannot render"):
        prompt.render(units()[0], language="python")


@pytest.mark.parametrize("content", ["user: only\n", "- a list\n", "system: [\n"])
def test_malformed_prompt_files_are_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(PromptError, match="not a prompt"):
        write_prompt(tmp_path, content)


def test_a_missing_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match=re.escape("v404.yaml")):
        PromptTemplate.load(tmp_path, "v404")
