"""
Contract-checked, partitioned Parquet storage for record tables.
"""

from assay.storage.parquet import ParquetStore, Partition
from assay.storage.schema import arrow_schema
from assay.storage.table import ContractError, TableSpec

__all__ = [
    "ContractError",
    "ParquetStore",
    "Partition",
    "TableSpec",
    "arrow_schema",
]
