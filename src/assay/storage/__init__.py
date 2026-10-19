"""
Contract-checked, partitioned Parquet storage for record tables.
"""

from assay.storage.schema import arrow_schema
from assay.storage.table import ContractError, TableSpec

__all__ = [
    "ContractError",
    "TableSpec",
    "arrow_schema",
]
