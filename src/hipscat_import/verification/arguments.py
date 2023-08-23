"""Utility to hold all arguments required throughout verification pipeline"""

from dataclasses import dataclass, field
from typing import List, Optional

from hipscat.catalog import Catalog

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class VerificationArguments(RuntimeArguments):
    """Data class for holding verification arguments"""

    ## Input
    input_catalog_path: str = ""
    input_catalog: Optional[Catalog] = None

    ## Verification options
    field_distribution_cols: List[str] = field(default_factory=list)
    """List of fields to get the overall distribution for. e.g. ["ra", "dec"].
    Should be valid columns in the parquet files."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        self.input_catalog = Catalog.read_from_hipscat(catalog_path=self.input_catalog_path)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "pipeline": "verification pipeline",
            "input_catalog_path": str(self.input_catalog_path),
            "field_distribution_cols": self.field_distribution_cols,
        }
