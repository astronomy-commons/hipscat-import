from dataclasses import dataclass
from typing import Any, Dict, Union

from hipscat.catalog.association_catalog.association_catalog import AssociationCatalogInfo
from hipscat.catalog.catalog_type import CatalogType
from hipscat.io.validation import is_valid_catalog

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class SoapArguments(RuntimeArguments):
    """Data class for holding source-object association arguments"""

    ## Input - Object catalog
    object_catalog_dir: str = ""
    object_id_column: str = ""
    object_storage_options: Union[Dict[Any, Any], None] = None
    """optional dictionary of abstract filesystem credentials for the OBJECT catalog."""

    ## Input - Source catalog
    source_catalog_dir: str = ""
    source_object_id_column: str = ""
    source_id_column: str = ""
    source_storage_options: Union[Dict[Any, Any], None] = None
    """optional dictionary of abstract filesystem credentials for the SOURCE catalog."""

    resume: bool = True
    """if there are existing intermediate resume files, should we
    read those and continue to run the pipeline where we left off"""
    write_leaf_files: bool = False
    """Should we also write out leaf parquet files (e.g. Norder/Dir/Npix.parquet)
    that represent the full association table"""

    compute_partition_size: int = 1_000_000_000

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.object_catalog_dir:
            raise ValueError("object_catalog_dir is required")
        if not self.object_id_column:
            raise ValueError("object_id_column is required")
        if not is_valid_catalog(self.object_catalog_dir, storage_options=self.object_storage_options):
            raise ValueError("object_catalog_dir not a valid catalog")

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")
        if not is_valid_catalog(self.source_catalog_dir, storage_options=self.source_storage_options):
            raise ValueError("source_catalog_dir not a valid catalog")

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_catalog_info(self, total_rows) -> AssociationCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": CatalogType.ASSOCIATION,
            "total_rows": total_rows,
            "primary_column": self.object_id_column,
            "primary_column_association": "object_id",
            "primary_catalog": self.object_catalog_dir,
            "join_column": self.source_object_id_column,
            "join_column_association": "source_id",
            "join_catalog": self.source_catalog_dir,
            "contains_leaf_files": self.write_leaf_files,
        }
        return AssociationCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "object_catalog_dir": self.object_catalog_dir,
            "object_id_column": self.object_id_column,
            "source_catalog_dir": self.source_catalog_dir,
            "source_object_id_column": self.source_object_id_column,
            "source_id_column": self.source_id_column,
            "compute_partition_size": self.compute_partition_size,
            "write_leaf_files": self.write_leaf_files,
        }
