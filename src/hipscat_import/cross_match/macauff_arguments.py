from __future__ import annotations

from dataclasses import dataclass, field
from os import path
from typing import List

from hipscat.catalog.association_catalog.association_catalog import AssociationCatalogInfo
from hipscat.catalog.catalog_type import CatalogType
from hipscat.io import FilePointer
from hipscat.io.validation import is_valid_catalog

from hipscat_import.catalog.file_readers import InputReader, get_file_reader
from hipscat_import.runtime_arguments import RuntimeArguments, find_input_paths

# pylint: disable=too-many-instance-attributes
# pylint: disable=unsupported-binary-operation


@dataclass
class MacauffArguments(RuntimeArguments):
    """Data class for holding cross-match association arguments"""

    ## Input - Cross-match data
    input_path: FilePointer | None = None
    """path to search for the input data"""
    input_format: str = ""
    """specifier of the input data format. this will be used to find an appropriate
    InputReader type, and may be used to find input files, via a match like
    ``<input_path>/*<input_format>`` """
    input_file_list: List[FilePointer] = field(default_factory=list)
    """can be used instead of `input_format` to import only specified files"""
    input_paths: List[FilePointer] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""

    ## Input - Left catalog
    left_catalog_dir: str = ""
    left_id_column: str = ""
    left_ra_column: str = ""
    left_dec_column: str = ""

    ## Input - Right catalog
    right_catalog_dir: str = ""
    right_id_column: str = ""
    right_ra_column: str = ""
    right_dec_column: str = ""

    ## `macauff` specific attributes
    metadata_file_path: str = ""
    resume: bool = True
    """if there are existing intermediate resume files, should we
    read those and continue to create a new catalog where we left off"""

    file_reader: InputReader | None = None

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()

        if not self.input_path and not self.input_file_list:
            raise ValueError("input_path nor input_file_list not provided")
        if not self.input_format:
            raise ValueError("input_format is required")

        if not self.left_catalog_dir:
            raise ValueError("left_catalog_dir is required")
        if not self.left_id_column:
            raise ValueError("left_id_column is required")
        if not self.left_ra_column:
            raise ValueError("left_ra_column is required")
        if not self.left_dec_column:
            raise ValueError("left_dec_column is required")
        if not is_valid_catalog(self.left_catalog_dir):
            raise ValueError("left_catalog_dir not a valid catalog")

        if not self.right_catalog_dir:
            raise ValueError("right_catalog_dir is required")
        if not self.right_id_column:
            raise ValueError("right_id_column is required")
        if not self.right_ra_column:
            raise ValueError("right_ra_column is required")
        if not self.right_dec_column:
            raise ValueError("right_dec_column is required")
        if not is_valid_catalog(self.right_catalog_dir):
            raise ValueError("right_catalog_dir not a valid catalog")

        if not self.metadata_file_path:
            raise ValueError("metadata_file_path required for macauff crossmatch")
        if not path.isfile(self.metadata_file_path):
            raise ValueError("Macauff column metadata file must point to valid file path.")

        # Basic checks complete - make more checks and create directories where necessary
        self.input_paths = find_input_paths(self.input_path, f"*{self.input_format}", self.input_file_list)

        if not self.file_reader:
            self.file_reader = get_file_reader(file_format=self.input_format)

    def to_catalog_info(self, total_rows) -> AssociationCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": CatalogType.ASSOCIATION,
            "total_rows": total_rows,
            "primary_column": self.left_id_column,
            "primary_catalog": str(self.left_catalog_dir),
            "join_column": self.right_id_column,
            "join_catalog": str(self.right_catalog_dir),
        }
        return AssociationCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "input_path": self.input_path,
            "input_format": self.input_format,
            "left_catalog_dir": self.left_catalog_dir,
            "left_id_column": self.left_id_column,
            "left_ra_column": self.left_ra_column,
            "left_dec_column": self.left_dec_column,
            "right_catalog_dir": self.right_catalog_dir,
            "right_id_column": self.right_id_column,
            "right_ra_column": self.right_ra_column,
            "right_dec_column": self.right_dec_column,
            "metadata_file_path": self.metadata_file_path,
        }
