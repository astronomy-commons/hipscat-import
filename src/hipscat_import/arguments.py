"""Utility to hold all arguments required throughout partitioning"""

from importlib.metadata import version

import pandas as pd
from hipscat.catalog import CatalogParameters
from hipscat.io import file_io

from hipscat_import.file_readers import get_file_reader

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


class ImportArguments:
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

    def __init__(
        self,
        catalog_name="",
        epoch="J2000",
        input_path="",
        input_format="parquet",
        input_file_list=None,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        add_hipscat_index=True,
        output_path="",
        overwrite=False,
        resume=False,
        highest_healpix_order=10,
        pixel_threshold=1_000_000,
        debug_stats_only=False,
        filter_function=None,
        file_reader=None,
        tmp_dir="",
        progress_bar=True,
        dask_tmp="",
        dask_n_workers=1,
        dask_threads_per_worker=1,
    ):
        self.catalog_name = catalog_name
        self.epoch = epoch
        self._input_path = file_io.get_file_pointer_from_path(input_path)
        self.input_format = input_format
        self._input_file_list = (
            [file_io.get_file_pointer_from_path(x) for x in input_file_list]
            if input_file_list
            else None
        )
        self.input_paths = []

        self.ra_column = ra_column
        self.dec_column = dec_column
        self.id_column = id_column
        self.add_hipscat_index = add_hipscat_index

        self._output_path = file_io.get_file_pointer_from_path(output_path)
        self.overwrite = overwrite
        self.resume = resume
        self.highest_healpix_order = highest_healpix_order
        self.pixel_threshold = pixel_threshold
        self.debug_stats_only = debug_stats_only
        self.catalog_path = ""
        self.tmp_path = ""

        self.filter_function = (
            filter_function if filter_function else passthrough_filter_function
        )
        self.file_reader = (
            file_reader if file_reader else get_file_reader(self.input_format)
        )

        self._tmp_dir = file_io.get_file_pointer_from_path(tmp_dir)
        self.progress_bar = progress_bar
        self.dask_tmp = file_io.get_file_pointer_from_path(dask_tmp)
        self.dask_n_workers = dask_n_workers
        self.dask_threads_per_worker = dask_threads_per_worker

        self._check_arguments()
        self._check_paths()

    def _check_arguments(self):
        """Check existence and consistency of argument values"""
        if not self.catalog_name:
            raise ValueError("catalog_name is required")
        if not self.input_format:
            raise ValueError("input_format is required")
        if not self._output_path:
            raise ValueError("output_path is required")

        if not 0 <= self.highest_healpix_order <= 10:
            raise ValueError("highest_healpix_order should be between 0 and 10")
        if not 100 <= self.pixel_threshold <= 1_000_000:
            raise ValueError("pixel_threshold should be between 100 and 1,000,000")

        if self.dask_n_workers <= 0:
            raise ValueError("dask_n_workers should be greather than 0")
        if self.dask_threads_per_worker <= 0:
            raise ValueError("dask_threads_per_worker should be greather than 0")

    def _check_paths(self):
        """Check existence and permissions on provided path arguments"""
        if (not self._input_path and not self._input_file_list) or (
            self._input_path and self._input_file_list
        ):
            raise ValueError("exactly one of input_path or input_file_list is required")

        if not file_io.does_file_or_directory_exist(self._output_path):
            raise FileNotFoundError(
                f"output_path ({self._output_path}) not found on local storage"
            )

        # Catalog path should not already exist, unless we're overwriting. Create it.
        self.catalog_path = file_io.append_paths_to_pointer(
            self._output_path, self.catalog_name
        )
        if not self.overwrite:
            if file_io.directory_has_contents(self.catalog_path):
                raise ValueError(
                    f"output_path ({self.catalog_path}) contains files."
                    " choose a different directory or use --overwrite flag"
                )
        file_io.make_directory(self.catalog_path, exist_ok=True)

        # Basic checks complete - make more checks and create directories where necessary
        if self._input_path:
            if not file_io.does_file_or_directory_exist(self._input_path):
                raise FileNotFoundError("input_path not found on local storage")
            self.input_paths = file_io.find_files_matching_path(
                self._input_path, f"*{self.input_format}"
            )

            if len(self.input_paths) == 0:
                raise FileNotFoundError(
                    f"No files matched file pattern: {self._input_path}*{self.input_format} "
                )
        elif self._input_file_list:
            self.input_paths = self._input_file_list
            for test_path in self.input_paths:
                if not file_io.does_file_or_directory_exist(test_path):
                    raise FileNotFoundError(f"{test_path} not found on local storage")
        if not self.input_paths:
            raise FileNotFoundError("No input files found")
        self.input_paths.sort()

        if self._tmp_dir:
            self.tmp_path = file_io.append_paths_to_pointer(
                self._tmp_dir, self.catalog_name, "intermediate"
            )
        elif self.dask_tmp:
            self.tmp_path = file_io.append_paths_to_pointer(
                self.dask_tmp, self.catalog_name, "intermediate"
            )
        else:
            self.tmp_path = file_io.append_paths_to_pointer(
                self.catalog_path, "intermediate"
            )
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
            catalog_name=self.catalog_name,
            input_paths=self.input_paths,
            input_format=self.input_format,
            output_path=self._output_path,
            highest_healpix_order=self.highest_healpix_order,
            pixel_threshold=self.pixel_threshold,
            epoch=self.epoch,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
        )

    def provenance_info(self) -> dict:
        """Fill all known information in a dictionary for provenance tracking.

        Returns:
            dictionary with all argument_name -> argument_value as key -> value pairs.
        """
        runtime_args = {
            "catalog_name": self.catalog_name,
            "epoch": self.epoch,
            "input_path": str(self._input_path),
            "input_paths": self.input_paths,
            "input_format": self.input_format,
            "input_file_list": self._input_file_list,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
            "id_column": self.id_column,
            "output_path": str(self._output_path),
            "overwrite": self.overwrite,
            "resume": self.resume,
            "highest_healpix_order": self.highest_healpix_order,
            "pixel_threshold": self.pixel_threshold,
            "debug_stats_only": self.debug_stats_only,
            "catalog_path": self.catalog_path,
            "tmp_path": str(self.tmp_path),
            "progress_bar": self.progress_bar,
            "dask_tmp": str(self.dask_tmp),
            "dask_n_workers": self.dask_n_workers,
            "dask_threads_per_worker": self.dask_threads_per_worker,
            "file_reader_info": self.file_reader.provenance_info(),
        }
        runtime_args["uses_filter_function"] = (
            self.filter_function != passthrough_filter_function
        )
        provenance_info = {
            "tool_name": "hipscat_import",
            "version": version("hipscat-import"),
            "runtime_args": runtime_args,
        }

        return provenance_info

    def __str__(self):
        formatted_string = (
            f"  catalog_name {self.catalog_name}\n"
            f"  input_path {self._input_path}\n"
            f"  input format {self.input_format}\n"
            f"  num input_paths {len(self.input_paths)}\n"
            f"  ra_column {self.ra_column}\n"
            f"  dec_column {self.dec_column}\n"
            f"  id_column {self.id_column}\n"
            f"  output_path {self._output_path}\n"
            f"  overwrite {self.overwrite}\n"
            f"  highest_healpix_order {self.highest_healpix_order}\n"
            f"  pixel_threshold {self.pixel_threshold}\n"
            f"  debug_stats_only {self.debug_stats_only}\n"
            f"  progress_bar {self.progress_bar}\n"
            f"  dask_tmp {self.dask_tmp}\n"
            f"  dask_n_workers {self.dask_n_workers}\n"
            f"  dask_threads_per_worker {self.dask_threads_per_worker}\n"
            f"  tmp_path {self.tmp_path}\n"
        )
        return formatted_string


def passthrough_filter_function(data: pd.DataFrame) -> pd.DataFrame:
    """No-op filter function to be used when no user-defined filter is provided"""
    return data
