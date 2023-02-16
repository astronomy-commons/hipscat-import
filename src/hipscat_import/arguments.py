"""Utility to hold all arguments required throughout partitioning"""

import glob
import os
import tempfile

import pandas as pd

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches


class ImportArguments:
    """Container class for holding partitioning arguments"""

    def __init__(
        self,
        catalog_name="",
        input_path="",
        input_format="parquet",
        input_file_list=None,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        output_path="",
        overwrite=False,
        highest_healpix_order=10,
        pixel_threshold=1_000_000,
        debug_stats_only=False,
        filter_function=None,
        schema_file=None,
        tmp_dir="",
        progress_bar=True,
        dask_tmp="",
        dask_n_workers=1,
        dask_threads_per_worker=1,
    ):
        self.catalog_name = catalog_name
        self.input_path = input_path
        self.input_format = input_format
        self.input_file_list = input_file_list
        self.input_paths = []

        self.ra_column = ra_column
        self.dec_column = dec_column
        self.id_column = id_column

        self.output_path = output_path
        self.overwrite = overwrite
        self.highest_healpix_order = highest_healpix_order
        self.pixel_threshold = pixel_threshold
        self.debug_stats_only = debug_stats_only
        self.catalog_path = ""

        self.filter_function = (
            filter_function if filter_function else passthrough_filter_function
        )
        self.schema_file = schema_file

        self.tmp_dir = tmp_dir
        self.progress_bar = progress_bar
        self.dask_tmp = dask_tmp
        self.dask_n_workers = dask_n_workers
        self.dask_threads_per_worker = dask_threads_per_worker

        # Any contexts that should be cleaned up on object deletion.
        self.contexts = []

        self._check_arguments()
        self._check_paths()

    def _check_arguments(self):
        """Check existence and consistency of argument values"""
        if not self.catalog_name:
            raise ValueError("catalog_name is required")
        if not self.input_format:
            raise ValueError("input_format is required")
        if not self.output_path:
            raise ValueError("output_path is required")

        if not 0 <= self.highest_healpix_order <= 10:
            raise ValueError("highest_healpix_order should be between 0 and 10")
        if not 100 <= self.pixel_threshold <= 1_000_000:
            raise ValueError("pixel_threshold should be between 0 and 1,000,000")

        if self.dask_n_workers <= 0:
            raise ValueError("dask_n_workers should be greather than 0")
        if self.dask_threads_per_worker <= 0:
            raise ValueError("dask_threads_per_worker should be greather than 0")

    def _check_paths(self):
        """Check existence and permissions on provided path arguments"""
        if (not self.input_path and not self.input_file_list) or (
            self.input_path and self.input_file_list
        ):
            raise ValueError("exactly one of input_path or input_file_list is required")

        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"output_path ({self.output_path}) not found on local storage"
            )

        # Catalog path should not already exist, unless we're overwriting. Create it.
        self.catalog_path = os.path.join(self.output_path, self.catalog_name)
        if not self.overwrite:
            existing_catalog_files = glob.glob(f"{self.catalog_path}/*")
            if existing_catalog_files:
                raise ValueError(
                    f"output_path ({self.catalog_path}) contains files."
                    " choose a different directory or use --overwrite flag"
                )
        os.makedirs(self.catalog_path, exist_ok=True)

        # Basic checks complete - make more checks and create directories where necessary
        if self.input_path:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError("input_path not found on local storage")
            self.input_paths = glob.glob(f"{self.input_path}/*{self.input_format}")
            if len(self.input_paths) == 0:
                raise FileNotFoundError(
                    f"No files matched file pattern: {self.input_path}*{self.input_format} "
                )
        elif self.input_file_list:
            self.input_paths = self.input_file_list
            for test_path in self.input_paths:
                if not os.path.exists(test_path):
                    raise FileNotFoundError(f"{test_path} not found on local storage")

        # Create a temp folder unique to this execution of the partitioner.
        # This avoids clobbering other executions.
        tmp_prefix = "/tmp"
        if self.tmp_dir:
            tmp_prefix = self.tmp_dir
        elif self.dask_tmp:
            tmp_prefix = self.dask_tmp
        else:
            tmp_prefix = self.output_path
        # pylint: disable=consider-using-with
        tmp_dir = tempfile.TemporaryDirectory(dir=tmp_prefix)
        self.tmp_dir = tmp_dir.name
        self.contexts.append(tmp_dir)

    def __str__(self):
        formatted_string = (
            f"  catalog_name {self.catalog_name}\n"
            f"  input_path {self.input_path}\n"
            f"  input format {self.input_format}\n"
            f"  num input_paths {len(self.input_paths)}\n"
            f"  ra_column {self.ra_column}\n"
            f"  dec_column {self.dec_column}\n"
            f"  id_column {self.id_column}\n"
            f"  output_path {self.output_path}\n"
            f"  overwrite {self.overwrite}\n"
            f"  highest_healpix_order {self.highest_healpix_order}\n"
            f"  pixel_threshold {self.pixel_threshold}\n"
            f"  debug_stats_only {self.debug_stats_only}\n"
            f"  progress_bar {self.progress_bar}\n"
            f"  dask_tmp {self.dask_tmp}\n"
            f"  dask_n_workers {self.dask_n_workers}\n"
            f"  dask_threads_per_worker {self.dask_threads_per_worker}\n"
            f"  tmp_dir {self.tmp_dir}\n"
        )
        return formatted_string


def passthrough_filter_function(data: pd.DataFrame) -> pd.DataFrame:
    """No-op filter function to be used when no user-defined filter is provided"""
    return data
