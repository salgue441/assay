"""
Dagster orchestration: partitioned assets for the pipeline's tables.

Import `assay.orchestrate.definitions` (not this package) to load the
Definitions; that module reads the config when imported.
"""

from assay.orchestrate.assets import (
    COMPLETE_CHECK,
    CONTRACT_CHECK,
    CORPUS_TABLES,
    corpus_assets,
    documents_assets,
    metrics_assets,
)
from assay.orchestrate.partitions import CorpusPartitions, partition_key
from assay.orchestrate.pipeline import (
    BUILD_CORPUS_JOB,
    DEFAULT_CONFIG,
    GENERATE_DOCUMENTS_JOB,
    MEASURE_DOCUMENTS_JOB,
    PartitionRun,
    Pipeline,
    build_pipeline,
)
from assay.orchestrate.resources import AssayResource

__all__ = [
    "BUILD_CORPUS_JOB",
    "COMPLETE_CHECK",
    "CONTRACT_CHECK",
    "CORPUS_TABLES",
    "DEFAULT_CONFIG",
    "GENERATE_DOCUMENTS_JOB",
    "MEASURE_DOCUMENTS_JOB",
    "AssayResource",
    "CorpusPartitions",
    "PartitionRun",
    "Pipeline",
    "build_pipeline",
    "corpus_assets",
    "documents_assets",
    "metrics_assets",
    "partition_key",
]
