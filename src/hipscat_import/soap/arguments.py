from dataclasses import dataclass

from hipscat.catalog.association_catalog.association_catalog import AssociationCatalogInfo
from hipscat.catalog.catalog_type import CatalogType

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class SoapArguments(RuntimeArguments):
    """Data class for holding source-object association arguments"""

    ## Input - Object catalog
    object_catalog_dir: str = ""
    object_id_column: str = ""

    ## Input - Source catalog
    source_catalog_dir: str = ""
    source_object_id_column: str = ""

    resume: bool = False
    """if there are existing intermediate resume files, should we
    read those and continue to run the pipeline where we left off"""

    compute_partition_size: int = 1_000_000_000

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.object_catalog_dir:
            raise ValueError("object_catalog_dir is required")
        if not self.object_id_column:
            raise ValueError("object_id_column is required")

        if not self.source_catalog_dir:
            raise ValueError("source_catalog_dir is required")
        if not self.source_object_id_column:
            raise ValueError("source_object_id_column is required")

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_catalog_info(self, total_rows) -> AssociationCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_catalog_name,
            "catalog_type": CatalogType.ASSOCIATION,
            "total_rows": total_rows,
            "primary_column": self.object_id_column,
            "primary_catalog": str(self.object_catalog_dir),
            "join_column": self.source_object_id_column,
            "join_catalog": str(self.source_catalog_dir),
        }
        return AssociationCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "object_catalog_dir": self.object_catalog_dir,
            "object_id_column": self.object_id_column,
            "source_catalog_dir": self.source_catalog_dir,
            "source_object_id_column": self.source_object_id_column,
            "compute_partition_size": self.compute_partition_size,
        }
