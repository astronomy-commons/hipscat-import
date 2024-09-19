"""Utility to hold all arguments required throughout verification pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hipscat.catalog import Catalog
from hipscat.io.validation import is_valid_catalog
from upath import UPath

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass(kw_only=True)
class VerificationArguments:
    """Data class for holding verification arguments"""

    ## Output
    output_path: str | Path | UPath = field()
    """Base path where verification reports should be written."""

    ## Input
    input_catalog_path: str | Path | UPath = field()
    """Path to an existing catalog that will be inspected."""

    ## Verification options
    use_schema_file: str | None = field(default=None)
    """Path to a parquet file containing the expected schema.
    Suggest to use the same value as when importing the catalog.
    If not provided, the catalog's _common_metadata file will be used as the source of truth.
    """
    expected_total_rows: int | None = field(default=None)
    """Total number of rows expected in this catalog."""
    field_distribution_cols: List[str] = field(default_factory=list)
    """List of fields to get the overall distribution for. e.g. ["ra", "dec"].
    Should be valid columns in the parquet files."""

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "pipeline": "verification pipeline",
            "input_catalog_path": self.input_catalog_path,
            "use_schema_file": self.use_schema_file,
            "expected_total_rows": self.expected_total_rows,
            "field_distribution_cols": self.field_distribution_cols,
        }
