from __future__ import annotations

from dataclasses import dataclass, field
from os import path
from typing import List

from hipscat.io import FilePointer, file_io
from hipscat.io.validation import is_valid_catalog

from hipscat_import.catalog.resume_plan import ResumePlan
from hipscat_import.runtime_arguments import RuntimeArguments

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
    add_hipscat_index: bool = True
    """add the hipscat spatial index field alongside the data"""
    highest_healpix_order: int = 10
    """healpix order to use when mapping. this will
    not necessarily be the order used in the final catalog, as we may combine
    pixels that don't meed the threshold"""
    pixel_threshold: int = 1_000_000
    """maximum number of rows for a single resulting pixel.
    we may combine hierarchically until we near the ``pixel_threshold``"""
    mapping_healpix_order: int = -1
    """healpix order to use when mapping. will be
    ``highest_healpix_order`` unless a positive value is provided for
    ``constant_healpix_order``"""

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
    match_probability_column: str = "match_p"
    column_names: List[str] = field(default_factory=list)

    resume: bool = True
    resume_plan: ResumePlan | None = None

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
        if self.input_path:
            if not file_io.does_file_or_directory_exist(self.input_path):
                raise FileNotFoundError("input_path not found on local storage")
            self.input_paths = file_io.find_files_matching_path(self.input_path, f"*{self.input_format}")
        elif self.input_file_list:
            self.input_paths = self.input_file_list
        if len(self.input_paths) == 0:
            raise FileNotFoundError("No input files found")

        self.column_names = self.get_column_names()

        self.resume_plan = ResumePlan(
            resume=self.resume,
            progress_bar=True,
            input_paths=self.input_paths,
            tmp_path=self.tmp_path,
        )

    def get_column_names(self):
        """Grab the macauff column names."""
        # TODO: Actually read in the metadata file once we get the example file from Tom.

        return [
            'Gaia_designation',
            'Gaia_RA',
            'Gaia_Dec',
            'BP',
            'G',
            'RP',
            'CatWISE_Name',
            'CatWISE_RA',
            'CatWISE_Dec',
            'W1',
            'W2',
            'match_p',
            'Separation',
            'eta',
            'xi',
            'Gaia_avg_cont',
            'CatWISE_avg_cont',
            'Gaia_cont_f1',
            'Gaia_cont_f10',
            'CatWISE_cont_f1',
            'CatWISE_cont_f10',
            'CatWISE_fit_sig',
        ]
