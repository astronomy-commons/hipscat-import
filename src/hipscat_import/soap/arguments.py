from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hipscat.catalog import Catalog
from hipscat.catalog.association_catalog.association_catalog import AssociationCatalogInfo
from hipscat.catalog.catalog_type import CatalogType
from hipscat.io.validation import is_valid_catalog
from upath import UPath

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class SoapArguments(RuntimeArguments):
    """Data class for holding source-object association arguments"""

    ## Input - Object catalog
    object_catalog_dir: str | Path | UPath | None = None
    object_id_column: str = ""

    ## Input - Source catalog
    source_catalog_dir: str | Path | UPath | None = None
    source_object_id_column: str = ""
    source_id_column: str = ""

    resume: bool = True
    """if there are existing intermediate resume files, should we
    read those and continue to run the pipeline where we left off"""
    delete_resume_log_files: bool = True
    """should we delete task-level done files once each stage is complete?
    if False, we will keep all done marker files at the end of the pipeline."""
    write_leaf_files: bool = False
    """Should we also write out leaf parquet files (e.g. Norder/Dir/Npix.parquet)
    that represent the full association table"""
    delete_intermediate_parquet_files: bool = True
    """should we delete the smaller intermediate parquet files generated in the
    mapping stage, once the relevant reducing stage is complete?"""

    compute_partition_size: int = 1_000_000_000

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.object_catalog_dir:
            raise ValueError("object_catalog_dir is required")
        if not self.object_id_column:
            raise ValueError("object_id_column is required")
        if not is_valid_catalog(self.object_catalog_dir):
            raise ValueError("object_catalog_dir not a valid catalog")

        self.object_catalog = Catalog.read_from_hipscat(catalog_path=self.object_catalog_dir)

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")
        if not is_valid_catalog(self.source_catalog_dir):
            raise ValueError("source_catalog_dir not a valid catalog")

        self.source_catalog = Catalog.read_from_hipscat(catalog_path=self.source_catalog_dir)

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
