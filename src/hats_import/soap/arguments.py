from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hats.catalog import Catalog, TableProperties
from hats.catalog.catalog_type import CatalogType
from hats.io.validation import is_valid_catalog
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


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
        if not is_valid_catalog(self.object_catalog_dir):
            raise ValueError("object_catalog_dir not a valid catalog")

        self.object_catalog = Catalog.read_hats(catalog_path=self.object_catalog_dir)

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")
        if not is_valid_catalog(self.source_catalog_dir):
            raise ValueError("source_catalog_dir not a valid catalog")

        self.source_catalog = Catalog.read_hats(catalog_path=self.source_catalog_dir)

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_table_properties(self, total_rows=10, highest_order=4, moc_sky_fraction=22 / 7) -> TableProperties:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": CatalogType.ASSOCIATION,
            "total_rows": total_rows,
            "primary_column": self.object_id_column,
            "primary_column_association": "object_id",
            "primary_catalog": str(self.object_catalog_dir),
            "join_column": self.source_object_id_column,
            "join_column_association": "source_id",
            "join_catalog": str(self.source_catalog_dir),
            "contains_leaf_files": self.write_leaf_files,
            "hats_order": highest_order,
            "moc_sky_fraction": f"{moc_sky_fraction:0.5f}",
        } | self.extra_property_dict()
        return TableProperties(**info)
