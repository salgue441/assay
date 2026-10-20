"""
Assemble the Dagster pipeline from a config file.

Both the Dagster entry point and the CLI build the pipeline here, so they
always run the same assets with the same resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dagster as dg

from assay.orchestrate.assets import (
    corpus_assets,
    documents_assets,
    judgments_assets,
    metrics_assets,
)
from assay.orchestrate.partitions import CorpusPartitions
from assay.orchestrate.resources import AssayResource
from assay.settings import load_settings

DEFAULT_CONFIG = Path("config/default.yaml")
BUILD_CORPUS_JOB = "build_corpus"
GENERATE_DOCUMENTS_JOB = "generate_documents"
MEASURE_DOCUMENTS_JOB = "measure_documents"
JUDGE_DOCUMENTS_JOB = "judge_documents"


@dataclass(frozen=True, slots=True)
class PartitionRun:
    """The outcome of materializing one asset for one partition."""

    key: str
    succeeded: bool
    """The run finished and every blocking (error-severity) check passed."""
    warnings: tuple[str, ...]
    """Non-blocking checks that failed, as `asset.check`."""
    metadata: dict[str, dict[str, int]]
    """Numeric metadata per materialized asset, such as stored `rows`."""
    error: str | None = None
    """Why the run failed, in one line, when it did."""


@dataclass(frozen=True)
class Pipeline:
    partitions: CorpusPartitions
    corpus: dg.AssetsDefinition
    documents: dg.AssetsDefinition
    metrics: dg.AssetsDefinition
    judgments: dg.AssetsDefinition
    resources: dict[str, AssayResource]

    @property
    def assets(self) -> list[dg.AssetsDefinition]:
        return [self.corpus, self.documents, self.metrics, self.judgments]

    def definitions(self) -> dg.Definitions:
        return dg.Definitions(
            assets=self.assets,
            jobs=[
                dg.define_asset_job(
                    BUILD_CORPUS_JOB,
                    selection=dg.AssetSelection.assets(self.corpus),
                    description="Fetch, extract and store the corpus tables of a repository epoch.",
                ),
                dg.define_asset_job(
                    GENERATE_DOCUMENTS_JOB,
                    selection=dg.AssetSelection.assets(self.documents),
                    description="Document a repository epoch's sampled units with each subject.",
                ),
                dg.define_asset_job(
                    MEASURE_DOCUMENTS_JOB,
                    selection=dg.AssetSelection.assets(self.metrics),
                    description="Compute the static metrics for a repository epoch's documents.",
                ),
                dg.define_asset_job(
                    JUDGE_DOCUMENTS_JOB,
                    selection=dg.AssetSelection.assets(self.judgments),
                    description="Grade a repository epoch's documents with every judge.",
                ),
            ],
            resources=self.resources,
        )

    def materialize_corpus(self, key: str, *, log_level: str = "INFO") -> PartitionRun:
        """Build and store one corpus partition in process, without a Dagster instance."""

        return self._materialize(self.corpus, key, log_level)

    def materialize_documents(self, key: str, *, log_level: str = "INFO") -> PartitionRun:
        """Generate one partition's documents; its corpus must already be stored."""

        return self._materialize(self.documents, key, log_level)

    def materialize_metrics(self, key: str, *, log_level: str = "INFO") -> PartitionRun:
        """Compute one partition's static metrics; its documents must already be stored."""

        return self._materialize(self.metrics, key, log_level)

    def materialize_judgments(self, key: str, *, log_level: str = "INFO") -> PartitionRun:
        """Grade one partition's documents; they must already be stored."""

        return self._materialize(self.judgments, key, log_level)

    def _materialize(self, asset: dg.AssetsDefinition, key: str, log_level: str) -> PartitionRun:
        result = dg.materialize(
            self.assets,
            selection=dg.AssetSelection.assets(asset),
            partition_key=key,
            resources=self.resources,
            raise_on_error=False,
            run_config={"loggers": {"console": {"config": {"log_level": log_level}}}},
        )
        checks = result.get_asset_check_evaluations()
        warnings = tuple(
            f"{check.asset_key.to_user_string()}.{check.check_name}"
            for check in checks
            if not check.passed and check.severity == dg.AssetCheckSeverity.WARN
        )
        metadata = {
            materialization.asset_key.to_user_string(): {
                name: int(value.value)  # type: ignore[arg-type]
                for name, value in materialization.metadata.items()
                if isinstance(value, dg.IntMetadataValue)
            }
            for materialization in result.asset_materializations_for_node(asset.node_def.name)
        }
        failed_checks = [
            f"{check.asset_key.to_user_string()}.{check.check_name} check failed"
            for check in checks
            if not check.passed and check.severity == dg.AssetCheckSeverity.ERROR
        ]
        errors = [*_step_errors(result), *failed_checks]
        return PartitionRun(
            key=key,
            succeeded=result.success and not failed_checks,
            warnings=warnings,
            metadata=metadata,
            error="; ".join(errors) or None,
        )


_REPORTED_AS_CHECKS = "DagsterAssetCheckFailedError"


def _step_errors(result: dg.ExecuteInProcessResult) -> list[str]:
    """
    One line per failed step: a `dg.Failure`'s own description, otherwise the
    first line of the exception as `Type: message`. Failures caused by
    blocking checks are skipped; the failed checks are reported instead.
    """

    messages = []
    for event in result.get_step_failure_events():
        failure = event.step_failure_data
        if failure.user_failure_data is not None and failure.user_failure_data.description:
            messages.append(failure.user_failure_data.description)
        elif failure.error is not None and failure.error.cls_name != _REPORTED_AS_CHECKS:
            error = failure.error
            # Dagster wraps the user's exception; the original is the cause.
            while error.cause is not None and (error.cls_name or "").startswith("Dagster"):
                error = error.cause
            first_line = error.message.strip().splitlines()[0]
            qualified_type, _, text = first_line.partition(": ")
            messages.append(f"{qualified_type.rsplit('.', 1)[-1]}: {text}")
    return messages


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
        partitions=partitions,
        corpus=corpus_assets(partitions),
        documents=documents_assets(partitions),
        metrics=metrics_assets(partitions),
        judgments=judgments_assets(partitions),
        resources={"assay": resource},
    )
