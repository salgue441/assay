"""A stored corpus with documents from two named subjects, for rating tests."""

from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from assay.corpus import SOURCES, UNITS, Snapshot
from assay.extract import SourceLanguage
from assay.generate import DOCUMENTS, Document, DocumentGenerator, PromptTemplate, Subject
from assay.llm import Provider
from assay.storage import ParquetStore, Partition

from ..fake_llm import FakeLLM
from ..generate.units import units

PROJECT_ROOT = Path(__file__).parents[2]
COMMIT = "c" * 40
SUBJECTS = ("writer-alpha", "writer-beta")
"""Distinctive names, so a test can prove they never reach the rater."""


def make_store(root: Path) -> tuple[ParquetStore, list[Document]]:
    store = ParquetStore(root)
    partition = Partition("owner/shapes", 0)
    unit_rows = units(commit=COMMIT)
    store.write(
        SOURCES,
        [
            Snapshot(
                repo="owner/shapes",
                epoch=0,
                commit=COMMIT,
                tree_hash="0" * 40,
                committed_at=datetime(2025, 10, 1, tzinfo=UTC),
                license="MIT",
                language=SourceLanguage.PYTHON,
                created_at=date(2025, 1, 1),
            )
        ],
        partition,
    )
    store.write(UNITS, unit_rows, partition)

    subjects = [
        Subject(
            name, Provider.OLLAMA, FakeLLM(model_id=f"{name}:1b", reply=lambda r: "## Doc\nText.")
        )
        for name in SUBJECTS
    ]
    prompt = PromptTemplate(version="v1", system="s", user="$name")
    documents = (
        DocumentGenerator(subjects, prompt, seed=1, max_output_tokens=64)
        .generate(unit_rows, language="python")
        .documents
    )
    store.write(DOCUMENTS, documents, partition)
    return store, documents


def write_config(root: Path) -> Path:
    config = root / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run": {"seed": 7},
                "storage": {"root": str(root / "data"), "duckdb_path": str(root / "a.duckdb")},
                "corpus": {
                    "manifest": str(root / "repos.yaml"),
                    "cache_dir": str(root / "cache"),
                    "allowed_licenses": ["MIT"],
                    "languages": ["python"],
                },
                "generation": {
                    "prompt_version": "v1",
                    "prompt_dir": str(PROJECT_ROOT / "config" / "prompts" / "generation"),
                    "subjects": [{"name": "w", "provider": "ollama", "model_id": "m"}],
                },
                "judge": {
                    "rubric_version": "v1",
                    "rubric_dir": str(PROJECT_ROOT / "config" / "rubrics"),
                    "runs_per_criterion": 1,
                    "criteria": ["accuracy"],
                    "variance_study": {"temperatures": [0.0], "subset_size": 1},
                    "models": [{"name": "j", "provider": "ollama", "model_id": "m"}],
                },
                "scoring": {
                    "weights": {"accuracy": 1.0},
                    "critical_failures": {"referential_integrity_below": 0.9},
                    "routing": {"publish_at_or_above": 80, "escalate_below": 50},
                },
            }
        ),
        encoding="utf-8",
    )
    return config
