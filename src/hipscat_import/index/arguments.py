"""Utility to hold all arguments required throughout indexing"""

from dataclasses import dataclass, field
from typing import List, Optional

from hipscat.catalog import Catalog
from hipscat.catalog.index.index_catalog_info import IndexCatalogInfo

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class IndexArguments(RuntimeArguments):
    """Data class for holding indexing arguments"""

    ## Input
    input_catalog_path: str = ""
    input_catalog: Optional[Catalog] = None
    indexing_column: str = ""
    extra_columns: List[str] = field(default_factory=list)

    ## Output
    include_hipscat_index: bool = True
    include_order_pixel: bool = True

    compute_partition_size: int = 1_000_000_000

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not self.indexing_column:
            raise ValueError("indexing_column is required")

        if not self.include_hipscat_index and not self.include_order_pixel:
            raise ValueError(
                "At least one of include_hipscat_index or include_order_pixel must be True"
            )

        self.input_catalog = Catalog.read_from_hipscat(
            catalog_path=self.input_catalog_path
        )

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_catalog_info(self, total_rows) -> IndexCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_catalog_name,
            "total_rows": total_rows,
            "catalog_type": "index",
            "primary_catalog": str(self.input_catalog_path),
            "indexing_column": self.indexing_column,
            "extra_columns": self.extra_columns,
        }
        return IndexCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "input_catalog_path": str(self.input_catalog_path),
            "indexing_column": self.indexing_column,
            "extra_columns": self.extra_columns,
            "include_hipscat_index": str(self.include_hipscat_index),
            "include_order_pixel": self.include_order_pixel,
        }
