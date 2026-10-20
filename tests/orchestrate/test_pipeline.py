import importlib
import sys
from pathlib import Path

import dagster as dg
import pytest

from assay.corpus import SOURCES, UNITS
from assay.orchestrate import (
    BUILD_CORPUS_JOB,
    CONTRACT_CHECK,
    GENERATE_DOCUMENTS_JOB,
    JUDGE_DOCUMENTS_JOB,
    MEASURE_DOCUMENTS_JOB,
    SCORE_DOCUMENTS_JOB,
    build_pipeline,
)
from assay.storage import ContractError, ParquetStore

from .project import Project

PROJECT_ROOT = Path(__file__).parents[2]


def test_materializes_and_checks_every_corpus_table(project: Project) -> None:
    pipeline = build_pipeline(project.config)

    run = pipeline.materialize_corpus("owner__greeter@01", log_level="WARNING")

    assert run.succeeded
    assert {table: counts["rows"] for table, counts in run.metadata.items()} == {
        "sources": 1,
        "units": 3,
        "symbols": 3,
    }
    store = ParquetStore(project.data)
    (snapshot,) = store.read(SOURCES)
    assert (snapshot.repo, snapshot.epoch, snapshot.commit) == (
        "owner/greeter",
        1,
        project.repo.commits[1],
    )
    assert {unit.qualified_name for unit in store.read(UNITS)} == {
        "greeter",
        "greeter.core",
        "greeter.core.greet",
    }


def test_contract_checks_run_for_each_table(project: Project) -> None:
    pipeline = build_pipeline(project.config)

    result = dg.materialize(
        [pipeline.corpus], partition_key="owner__greeter@00", resources=pipeline.resources
    )

    evaluations = result.get_asset_check_evaluations()
    assert {(e.asset_key.to_user_string(), e.check_name) for e in evaluations} == {
        ("sources", CONTRACT_CHECK),
        ("units", CONTRACT_CHECK),
        ("symbols", CONTRACT_CHECK),
    }
    assert all(e.passed for e in evaluations)


def test_a_stored_partition_that_breaks_its_contract_fails_the_run(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_read(*_: object, **__: object) -> list[object]:
        msg = "units: contract violated"
        raise ContractError(msg)

    monkeypatch.setattr(ParquetStore, "read", broken_read)

    run = build_pipeline(project.config).materialize_corpus("owner__greeter@00", log_level="ERROR")

    assert not run.succeeded
    assert all("rows" not in counts for counts in run.metadata.values())
    assert run.error == "; ".join(
        f"{table}.contract check failed" for table in ("sources", "units", "symbols")
    )


def test_definitions_load_with_a_job_per_stage(project: Project) -> None:
    defs = build_pipeline(project.config).definitions()

    for name in (
        BUILD_CORPUS_JOB,
        GENERATE_DOCUMENTS_JOB,
        MEASURE_DOCUMENTS_JOB,
        JUDGE_DOCUMENTS_JOB,
        SCORE_DOCUMENTS_JOB,
    ):
        job = defs.resolve_job_def(name)
        assert job.partitions_def is not None
        assert job.partitions_def.get_partition_keys() == ["owner__greeter@00", "owner__greeter@01"]
    assert {key.to_user_string() for key in defs.resolve_all_asset_keys()} == {
        "sources",
        "units",
        "symbols",
        "documents",
        "metrics",
        "judgments",
        "scores",
    }


def test_the_dagster_entry_point_loads_the_committed_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.delenv("ASSAY_CONFIG_FILE", raising=False)
    sys.modules.pop("assay.orchestrate.definitions", None)

    module = importlib.import_module("assay.orchestrate.definitions")

    assert (
        "pallets__click@00"
        in module.defs.resolve_job_def(BUILD_CORPUS_JOB).partitions_def.get_partition_keys()
    )


def test_an_unexpected_error_is_reported_as_type_and_message(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_write(*_: object, **__: object) -> None:
        msg = "disk full. Free 2.5 GB and retry"
        raise OSError(msg)

    monkeypatch.setattr(ParquetStore, "write", broken_write)

    run = build_pipeline(project.config).materialize_corpus(
        "owner__greeter@00", log_level="CRITICAL"
    )

    assert not run.succeeded
    assert run.error == "OSError: disk full. Free 2.5 GB and retry"
