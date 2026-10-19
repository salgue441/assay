import os
from pathlib import Path, PurePosixPath

import pytest

from assay.corpus import PythonLayout, discover_sources


def write(root: Path, files: dict[str, str | bytes]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8", newline="")


def discovered(root: Path, **kwargs: object) -> list[str]:
    return [str(s.path) for s in discover_sources(root, PythonLayout(), **kwargs)]  # type: ignore[arg-type]


def test_finds_source_in_sorted_order_with_module_names(tmp_path: Path) -> None:
    write(tmp_path, {"src/pkg/b.py": "", "src/pkg/a.py": "", "src/pkg/__init__.py": ""})

    sources = list(discover_sources(tmp_path, PythonLayout()))

    assert [str(s.path) for s in sources] == [
        "src/pkg/__init__.py",
        "src/pkg/a.py",
        "src/pkg/b.py",
    ]
    assert [s.module for s in sources] == ["pkg", "pkg.a", "pkg.b"]
    assert all(isinstance(s.path, PurePosixPath) for s in sources)


def test_skips_tests_hidden_and_excluded_directories(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "pkg/core.py": "",
            "pkg/test_core.py": "",
            "tests/test_all.py": "",
            "pkg/tests/helpers.py": "",
            "docs/conf.py": "",
            ".github/tool.py": "",
            "pkg/.cache/x.py": "",
        },
    )

    assert discovered(tmp_path) == ["pkg/core.py"]


def test_excluded_directories_are_configurable(tmp_path: Path) -> None:
    write(tmp_path, {"pkg/core.py": "", "pkg/legacy/old.py": ""})

    assert discovered(tmp_path, excluded_dirs={"legacy"}) == ["pkg/core.py"]


def test_skips_oversized_non_utf8_and_unnameable_files(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "pkg/ok.py": "x = 1\n",
            "pkg/huge.py": "x" * 2_000,
            "pkg/latin1.py": "s = 'caf\xe9'\n".encode("latin-1"),
            "bad-name/tool.py": "",
        },
    )

    assert discovered(tmp_path, max_file_bytes=1_000) == ["pkg/ok.py"]


def test_normalizes_line_endings(tmp_path: Path) -> None:
    write(tmp_path, {"pkg/mod.py": "a = 1\r\nb = 2\r\n"})

    (source,) = discover_sources(tmp_path, PythonLayout())

    assert source.text == "a = 1\nb = 2\n"


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs privileges on Windows")
def test_skips_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n", encoding="utf-8")
    write(tmp_path / "repo", {"pkg/real.py": ""})
    (tmp_path / "repo" / "pkg" / "link.py").symlink_to(outside)

    assert discovered(tmp_path / "repo") == ["pkg/real.py"]
