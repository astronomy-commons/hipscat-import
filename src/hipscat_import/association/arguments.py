"""Utility to hold all arguments required throughout association pipeline"""

from dataclasses import dataclass

from hipscat.catalog import CatalogParameters

from hipscat_import.runtime_arguments import RuntimeArguments

# pylint: disable=too-many-instance-attributes


@dataclass
class AssociationArguments(RuntimeArguments):
    """Data class for holding association arguments"""

    ## Input - Primary
    primary_input_catalog_path: str = ""
    primary_id_column: str = ""
    primary_join_column: str = ""

    ## Input - Join
    join_input_catalog_path: str = ""
    join_id_column: str = ""
    join_foreign_key: str = ""

    join_how: str = "inner"
    compute_partition_size: int = 1_000_000_000

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.primary_input_catalog_path:
            raise ValueError("primary_input_catalog_path is required")
        if not self.primary_id_column:
            raise ValueError("primary_id_column is required")
        if not self.primary_join_column:
            raise ValueError("primary_join_column is required")
        if self.primary_id_column in ["primary_id", "join_id"]:
            raise ValueError("primary_id_column uses a reserved column name")

        if not self.join_input_catalog_path:
            raise ValueError("join_input_catalog_path is required")
        if not self.join_id_column:
            raise ValueError("join_id_column is required")
        if not self.join_foreign_key:
            raise ValueError("join_foreign_key is required")
        if self.join_id_column in ["primary_id", "join_id"]:
            raise ValueError("join_id_column uses a reserved column name")

        if self.join_how not in ["left", "right", "outer", "inner"]:
            raise ValueError("join_how must be one of left, right, outer, or inner")

        if self.compute_partition_size < 100_000:
            raise ValueError("compute_partition_size must be at least 100_000")

    def to_catalog_parameters(self) -> CatalogParameters:
        """Convert importing arguments into hipscat catalog parameters.

        Returns:
            CatalogParameters for catalog being created.
        """
        return CatalogParameters(
            catalog_name=self.output_catalog_name,
            catalog_type="association",
            output_path=self.output_path,
        )

    def additional_runtime_provenance_info(self):
        return {
            "primary_input_catalog_path": str(self.primary_input_catalog_path),
            "primary_id_column": self.primary_id_column,
            "primary_join_column": self.primary_join_column,
            "join_input_catalog_path": str(self.join_input_catalog_path),
            "join_id_column": self.join_id_column,
            "join_foreign_key": self.join_foreign_key,
        }
