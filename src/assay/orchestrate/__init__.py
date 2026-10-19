"""
Dagster orchestration: partitioned assets for the pipeline's tables.

Import `assay.orchestrate.definitions` (not this package) to load the
Definitions; that module reads the config when imported.
"""

from assay.orchestrate.assets import CONTRACT_CHECK, CORPUS_TABLES, corpus_assets
from assay.orchestrate.partitions import CorpusPartitions, partition_key
from assay.orchestrate.pipeline import BUILD_CORPUS_JOB, DEFAULT_CONFIG, Pipeline, build_pipeline
from assay.orchestrate.resources import AssayResource

__all__ = [
    "BUILD_CORPUS_JOB",
    "CONTRACT_CHECK",
    "CORPUS_TABLES",
    "DEFAULT_CONFIG",
    "AssayResource",
    "CorpusPartitions",
    "Pipeline",
    "build_pipeline",
    "corpus_assets",
    "partition_key",
]
