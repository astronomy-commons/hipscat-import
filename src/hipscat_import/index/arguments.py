"""Utility to hold all arguments required throughout indexing"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from hipscat.catalog import Catalog
from hipscat.catalog.index.index_catalog_info import IndexCatalogInfo
from hipscat.io.validation import is_valid_catalog

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class IndexArguments(RuntimeArguments):
    """Data class for holding indexing arguments"""

    ## Input
    input_catalog_path: str = ""
    input_catalog: Optional[Catalog] = None
    input_storage_options: Union[Dict[Any, Any], None] = None
    """optional dictionary of abstract filesystem credentials for the INPUT."""
    indexing_column: str = ""
    extra_columns: List[str] = field(default_factory=list)

    ## Output
    include_hipscat_index: bool = True
    """Include the hipscat spatial partition index."""
    include_order_pixel: bool = True
    """Include partitioning columns, Norder, Dir, and Npix. You probably want to keep these!"""
    drop_duplicates: bool = True
    """Should we check for duplicate rows (including new indexing column),
    and remove duplicates before writing to new index catalog?
    If you know that your data will not have duplicates (e.g. an index over
    a unique primary key), set to False to avoid unnecessary work."""

    ## Compute parameters
    compute_partition_size: int = 1_000_000_000
    """partition size used when creating leaf parquet files."""
    division_hints: Optional[List] = None
    """Hints used when splitting up the rows by the new index. If you have
    some prior knowledge about the distribution of your indexing_column, 
    providing it here can speed up calculations dramatically. Note that
    these will NOT necessarily be the divisions that the data is partitioned
    along."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not self.indexing_column:
            raise ValueError("indexing_column is required")

        if not self.include_hipscat_index and not self.include_order_pixel:
            raise ValueError("At least one of include_hipscat_index or include_order_pixel must be True")

        if not is_valid_catalog(self.input_catalog_path, storage_options=self.input_storage_options):
            raise ValueError("input_catalog_path not a valid catalog")
        self.input_catalog = Catalog.read_from_hipscat(
            catalog_path=self.input_catalog_path, storage_options=self.input_storage_options
        )

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_catalog_info(self, total_rows) -> IndexCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "total_rows": total_rows,
            "catalog_type": "index",
            "primary_catalog": self.input_catalog_path,
            "indexing_column": self.indexing_column,
            "extra_columns": self.extra_columns,
        }
        return IndexCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "input_catalog_path": self.input_catalog_path,
            "indexing_column": self.indexing_column,
            "extra_columns": self.extra_columns,
            "include_hipscat_index": self.include_hipscat_index,
            "include_order_pixel": self.include_order_pixel,
        }
