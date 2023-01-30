"""Utility to hold all arguments required throughout partitioning"""
import argparse
import glob
import os
import tempfile

import pandas as pd


class ImportArguments:
    """Container class for holding partitioning arguments"""

    def __init__(self):
        self.catalog_name = ""
        self.input_path = ""
        self.input_format = ""
        self.input_file_list = []
        self.input_paths = []

        self.ra_column = ""
        self.dec_column = ""
        self.id_column = ""

        self.output_path = ""
        self.catalog_path = ""
        self.overwrite = False
        self.highest_healpix_order = 10
        self.pixel_threshold = 1_000_000
        self.debug_stats_only = False

        self.filter_function = None

        self.progress_bar = True
        self.dask_tmp = ""
        self.dask_n_workers = 1
        self.dask_threads_per_worker = 1

        self.tmp_dir = ""
        # Any contexts that should be cleaned up on object deletion.
        self.contexts = []

    def from_command_line(self, cl_args):
        """Parse arguments from the command line"""

        parser = argparse.ArgumentParser(
            prog="LSD2 Partitioner",
            description="Instantiate a partitioned catalog from unpartitioned sources",
        )

        # ===========           INPUT ARGUMENTS           ===========
        group = parser.add_argument_group("INPUT")
        group.add_argument(
            "-c",
            "--catalog_name",
            help="short name for the catalog that will be used for the output directory",
            default=None,
            type=str,
        )
        group.add_argument(
            "-i",
            "--input_path",
            help="path prefix for unpartitioned input files",
            default=None,
            type=str,
        )
        group.add_argument(
            "-fmt",
            "--input_format",
            help="file format for unpartitioned input files",
            default="parquet",
            type=str,
        )
        group.add_argument(
            "--input_file_list",
            help="explicit list of input files, comma-separated",
            default="",
            type=str,
        )

        # ===========            INPUT COLUMNS            ===========
        group = parser.add_argument_group(
            "INPUT COLUMNS",
            """Column names in the input source that
            correspond to spatial attributes used in partitioning""",
        )
        group.add_argument(
            "-ra",
            "--ra_column",
            help="column name for the ra (rate of ascension)",
            default="ra",
            type=str,
        )
        group.add_argument(
            "-dec",
            "--dec_column",
            help="column name for the dec (declination)",
            default="dec",
            type=str,
        )
        group.add_argument(
            "-id",
            "--id_column",
            help="column name for the object id",
            default="id",
            type=str,
        )
        # ===========           OUTPUT ARGUMENTS          ===========
        group = parser.add_argument_group("OUTPUT")
        group.add_argument(
            "-o",
            "--output_path",
            help="path prefix for partitioned output and metadata files",
            default=None,
            type=str,
        )
        group.add_argument(
            "--overwrite",
            help="if set, the any existing catalog data will be overwritten",
            action="store_true",
        )
        group.add_argument(
            "--no_overwrite",
            help="if set, the pipeline will exit if existing output is found",
            dest="overwrite",
            action="store_false",
        )

        # ===========           STATS ARGUMENTS           ===========
        group = parser.add_argument_group("STATS")
        group.add_argument(
            "-ho",
            "--highest_healpix_order",
            help="the most dense healpix order (7-10 is a good range for this)",
            default=10,
            type=int,
        )
        group.add_argument(
            "-pt",
            "--pixel_threshold",
            help="maximum objects allowed in a single pixel",
            default=1_000_000,
            type=int,
        )
        group.add_argument(
            "--debug_stats_only",
            help="""DEBUGGING FLAG -
            if set, the pipeline will only fetch statistics about the origin data
            and will not generate partitioned output""",
            action="store_true",
        )
        group.add_argument(
            "--no_debug_stats_only",
            help="DEBUGGING FLAG - if set, the pipeline will generate partitioned output",
            dest="debug_stats_only",
            action="store_false",
        )
        # ===========         EXECUTION ARGUMENTS         ===========
        group = parser.add_argument_group("EXECUTION")
        group.add_argument(
            "--progress_bar",
            help="should a progress bar be displayed?",
            default=True,
            action="store_true",
        )
        group.add_argument(
            "--no_progress_bar",
            help="should a progress bar be displayed?",
            dest="progress_bar",
            action="store_false",
        )
        group.add_argument(
            "--tmp_dir",
            help="directory for storing temporary parquet files",
            default=None,
            type=str,
        )
        group.add_argument(
            "-dt",
            "--dask_tmp",
            help="directory for storing temporary files generated by dask engine",
            default=None,
            type=str,
        )
        group.add_argument(
            "--dask_n_workers",
            help="the number of dask workers available",
            default=1,
            type=int,
        )
        group.add_argument(
            "--dask_threads_per_worker",
            help="the number of threads per dask worker",
            default=1,
            type=int,
        )

        args = parser.parse_args(cl_args)

        self.from_params(
            catalog_name=args.catalog_name,
            input_path=args.input_path,
            input_format=args.input_format,
            input_file_list=(
                args.input_file_list.split(",") if args.input_file_list else None
            ),
            ra_column=args.ra_column,
            dec_column=args.dec_column,
            id_column=args.id_column,
            output_path=args.output_path,
            overwrite=args.overwrite,
            highest_healpix_order=args.highest_healpix_order,
            pixel_threshold=args.pixel_threshold,
            debug_stats_only=args.debug_stats_only,
            tmp_dir=args.tmp_dir,
            progress_bar=args.progress_bar,
            dask_tmp=args.dask_tmp,
            dask_n_workers=args.dask_n_workers,
            dask_threads_per_worker=args.dask_threads_per_worker,
        )

    def from_params(
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
        tmp_dir="",
        progress_bar=True,
        dask_tmp="",
        dask_n_workers=1,
        dask_threads_per_worker=1,
    ):
        """Use arguments provided in parameters."""
        self.catalog_name = catalog_name
        self.input_path = input_path
        self.input_format = input_format
        self.input_file_list = input_file_list

        self.ra_column = ra_column
        self.dec_column = dec_column
        self.id_column = id_column

        self.output_path = output_path
        self.overwrite = overwrite
        self.highest_healpix_order = highest_healpix_order
        self.pixel_threshold = pixel_threshold
        self.debug_stats_only = debug_stats_only

        self.filter_function = (
            filter_function if filter_function else passthrough_filter_function
        )

        self.tmp_dir = tmp_dir
        self.progress_bar = progress_bar
        self.dask_tmp = dask_tmp
        self.dask_n_workers = dask_n_workers
        self.dask_threads_per_worker = dask_threads_per_worker

        self.check_arguments()
        self.check_paths()

    def check_arguments(self):
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

    def check_paths(self):
        """Check existence and permissions on provided path arguments"""
        # TODO: handle non-posix files/paths
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
