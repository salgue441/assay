"""
Dagster orchestration: partitioned assets for the pipeline's tables.

Import `assay.orchestrate.definitions` (not this package) to load the
Definitions; that module reads the config when imported.
"""

from assay.orchestrate.partitions import CorpusPartitions, partition_key

__all__ = [
    "CorpusPartitions",
    "partition_key",
]
