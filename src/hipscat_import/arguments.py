"""Utility to hold all arguments required throughout partitioning"""

from importlib.metadata import version

import pandas as pd
from hipscat.catalog import CatalogParameters
from hipscat.io import file_io

from hipscat_import.file_readers import get_file_reader

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


class ImportArguments:
    """Container class for holding partitioning arguments"""

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
