"""
Assemble the Dagster pipeline from a config file.

Both the Dagster entry point and the CLI build the pipeline here, so they
always run the same assets with the same resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dagster as dg

from assay.orchestrate.assets import corpus_assets
from assay.orchestrate.partitions import CorpusPartitions
from assay.orchestrate.resources import AssayResource
from assay.settings import load_settings

DEFAULT_CONFIG = Path("config/default.yaml")
BUILD_CORPUS_JOB = "build_corpus"


@dataclass(frozen=True)
class Pipeline:
    partitions: CorpusPartitions
    corpus: dg.AssetsDefinition
    resources: dict[str, AssayResource]

    def definitions(self) -> dg.Definitions:
        return dg.Definitions(
            assets=[self.corpus],
            jobs=[
                dg.define_asset_job(
                    BUILD_CORPUS_JOB,
                    selection=dg.AssetSelection.assets(self.corpus),
                    description="Fetch, extract and store the corpus tables of a repository epoch.",
                )
            ],
            resources=self.resources,
        )

    def materialize_corpus(self, key: str, *, log_level: str = "INFO") -> CorpusRun:
        """Build and store one partition in process, without a Dagster instance."""

        result = dg.materialize(
            [self.corpus],
            partition_key=key,
            resources=self.resources,
            raise_on_error=False,
            run_config={"loggers": {"console": {"config": {"log_level": log_level}}}},
        )
        rows = {
            materialization.asset_key.to_user_string(): int(materialization.metadata["rows"].value)  # type: ignore[arg-type]
            for materialization in result.asset_materializations_for_node(self.corpus.node_def.name)
            if "rows" in materialization.metadata
        }
        return CorpusRun(
            key=key,
            succeeded=result.success
            and all(check.passed for check in result.get_asset_check_evaluations()),
            rows=rows,
        )


@dataclass(frozen=True, slots=True)
class CorpusRun:
    """The outcome of building one corpus partition."""

    key: str
    succeeded: bool
    """The run finished and every contract check passed."""
    rows: dict[str, int]
    """Stored rows per table that passed its contract check."""


def build_pipeline(config_path: Path = DEFAULT_CONFIG, *, base_dir: Path | None = None) -> Pipeline:
    """
    Load and check the config and manifest, and assemble the pipeline.

    Raises `SettingsError` or `ManifestError` for an invalid config or manifest.
    """

    settings = load_settings(config_path, base_dir=base_dir)
    partitions = CorpusPartitions.from_manifest(settings.corpus.load_manifest())
    resource = AssayResource(
        config_path=str(config_path), base_dir=None if base_dir is None else str(base_dir)
    )
    return Pipeline(
        partitions=partitions, corpus=corpus_assets(partitions), resources={"assay": resource}
    )
