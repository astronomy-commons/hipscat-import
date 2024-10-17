"""Utility to hold all arguments required throughout verification pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hats.catalog import Catalog
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class VerificationArguments(RuntimeArguments):
    """Data class for holding verification arguments"""

    ## Input
    input_catalog_path: str | Path | UPath | None = None
    """Path to an existing catalog that will be inspected."""
    input_catalog: Optional[Catalog] = None
    """In-memory representation of a catalog. If not provided, it will be loaded
    from the input_catalog_path."""

    ## Verification options
    field_distribution_cols: List[str] = field(default_factory=list)
    """List of fields to get the overall distribution for. e.g. ["ra", "dec"].
    Should be valid columns in the parquet files."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path and not self.input_catalog:
            raise ValueError("input catalog is required (either input_catalog_path or input_catalog)")
        if not self.input_catalog:
            if not is_valid_catalog(self.input_catalog_path):
                raise ValueError("input_catalog_path not a valid catalog")
            self.input_catalog = Catalog.read_hats(catalog_path=self.input_catalog_path)
        if not self.input_catalog_path:
            self.input_catalog_path = self.input_catalog.catalog_path
