"""Utility to hold all arguments required throughout partitioning"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd
from hipscat.catalog import CatalogParameters
from hipscat.io import file_io

from hipscat_import.catalog.file_readers import InputReader, get_file_reader
from hipscat_import.runtime_arguments import RuntimeArguments

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


@dataclass
class ImportArguments(RuntimeArguments):
    """Container class for holding partitioning arguments


    Attributes:
        catalog_name (str): short, convenient name for the catalog.
        epoch (str): astronomical epoch for the data. defaults to "J2000"
        `_input_path` (str): path to the input data
        input_format (str): specifier of the input data format. This will
            be used to find an appropriate file reader, and may be used to find
            input files, via a match like "<input_path>/*<input_format>"
        `_input_file_list` (list[str]): can be used instead of `input_format` to
            import just a few files at a time.
        input_paths (list[str]): resolved list of all files used in the importer
        ra_column (str): column for right ascension
        dec_column (str): column for declination
        id_column (str): column for survey identifier, or other sortable column
        `_output_path` (str): base path for catalog output
        catalog_path (str): path for catalog output. This is generally formed
            via <output_path>/<catalog_name>
        overwrite (bool): if there is existing data at the `catalog_path`, should
            we overwrite and create a new catalog.
        resume (bool): if there are existing intermediate resume files, should we
            read those and continue to create a new catalog where we left off.
        highest_healpix_order (int): healpix order to use when mapping. this will
            not necessarily be the order used in the final catalog, as we may combine
            pixels that don't meed the threshold.
        pixel_threshold (int): maximum number of rows for a single resulting pixel.
            we may combine hierarchically until we near the `pixel_threshold`
        debug_stats_only (bool): do not perform a map reduce and don't create a new
            catalog. generate the partition info.
        tmp_path (str): path for storing intermediate files
        filter_function (function pointer): optional method which takes a pandas
            dataframe as input, performs some filtering or transformation of the data,
            and returns a dataframe with the rows that will be used to create the
            new catalog.
        file_reader (`InputReader`): instance of input reader that specifies arguments
            necessary for reading from your input files.
        `_tmp_dir` (str): base directory provided by the caller for temporary files.
        progress_bar (bool): if true, a tqdm progress bar will be displayed for user
            feedback of map reduce progress.
        dask_tmp (str): directory for dask worker space. this should be local to
            the execution of the pipeline, for speed of reads and writes.
        dask_n_workers (int): number of workers for the dask client
        dask_threads_per_worker (int): number of threads per dask worker.
    """

    epoch: str = "J2000"
    catalog_type: str = "object"
    input_path: str = ""
    input_format: str = ""
    input_file_list: List[str] = field(default_factory=list)

    ra_column: str = "ra"
    dec_column: str = "dec"
    id_column: str = "id"
    add_hipscat_index: bool = True
    use_schema_file: str | None = None
    resume: bool = False
    highest_healpix_order: int = 10
    pixel_threshold: int = 1_000_000
    debug_stats_only: bool = False
    filter_function: Callable | None = None
    file_reader: InputReader | None = None

    def __post_init__(self):
        RuntimeArguments._check_arguments(self)
        self.input_path = file_io.get_file_pointer_from_path(self.input_path)
        self.input_file_list = (
            [file_io.get_file_pointer_from_path(x) for x in self.input_file_list]
            if self.input_file_list
            else None
        )
        self.input_paths = []

        if not self.filter_function:
            self.filter_function = passthrough_filter_function

        self._check_arguments()

    def _check_arguments(self):
        """Check existence and consistency of argument values"""
        if not self.input_format:
            raise ValueError("input_format is required")

        if not 0 <= self.highest_healpix_order <= 19:
            raise ValueError("highest_healpix_order should be between 0 and 19")
        if not 100 <= self.pixel_threshold <= 10_000_000:
            raise ValueError("pixel_threshold should be between 100 and 10,000,000")

        if self.catalog_type not in ("source", "object"):
            raise ValueError("catalog_type should be one of `source` or `object`")

        if (not self.input_path and not self.input_file_list) or (
            self.input_path and self.input_file_list
        ):
            raise ValueError("exactly one of input_path or input_file_list is required")
        if not self.file_reader:
            self.file_reader = get_file_reader(self.input_format)

        # Basic checks complete - make more checks and create directories where necessary
        if self.input_path:
            if not file_io.does_file_or_directory_exist(self.input_path):
                raise FileNotFoundError("input_path not found on local storage")
            self.input_paths = file_io.find_files_matching_path(
                self.input_path, f"*{self.input_format}"
            )

            if len(self.input_paths) == 0:
                raise FileNotFoundError(
                    f"No files matched file pattern: {self.input_path}*{self.input_format} "
                )
        elif self.input_file_list:
            self.input_paths = self.input_file_list
            for test_path in self.input_paths:
                if not file_io.does_file_or_directory_exist(test_path):
                    raise FileNotFoundError(f"{test_path} not found on local storage")
        self.input_paths.sort()

        if not self.resume:
            if file_io.directory_has_contents(self.tmp_path):
                raise ValueError(
                    f"tmp_path ({self.tmp_path}) contains intermediate files."
                    " choose a different directory or use --resume flag"
                )
        file_io.make_directory(self.tmp_path, exist_ok=True)

    def to_catalog_parameters(self) -> CatalogParameters:
        """Convert importing arguments into hipscat catalog parameters.

        Returns:
            CatalogParameters for catalog being created.
        """
        return CatalogParameters(
            catalog_name=self.output_catalog_name,
            catalog_type=self.catalog_type,
            output_path=self.output_path,
            epoch=self.epoch,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
        )

    def additional_provenance_info(self):
        return {
            "catalog_name": self.output_catalog_name,
            "epoch": self.epoch,
            "catalog_type": self.catalog_type,
            "input_path": str(self.input_path),
            "input_paths": self.input_paths,
            "input_format": self.input_format,
            "input_file_list": self.input_file_list,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
            "id_column": self.id_column,
            "highest_healpix_order": self.highest_healpix_order,
            "pixel_threshold": self.pixel_threshold,
            "debug_stats_only": self.debug_stats_only,
            "file_reader_info": self.file_reader.provenance_info(),
        }


def passthrough_filter_function(data: pd.DataFrame) -> pd.DataFrame:
    """No-op filter function to be used when no user-defined filter is provided"""
    return data
