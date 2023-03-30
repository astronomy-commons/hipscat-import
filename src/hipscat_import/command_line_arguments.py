"""Parse import arguments from command line"""

import argparse

from hipscat_import.arguments import ImportArguments
from hipscat_import.file_readers import get_file_reader


def parse_command_line(cl_args):
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

    # ===========           READER ARGUMENTS           ===========
    group = parser.add_argument_group("READER")
    group.add_argument(
        "--schema_file",
        help="parquet file that contains field names and types to be used when reading a CSV file",
        default=None,
        type=str,
    )
    group.add_argument(
        "--header_rows",
        help="number of rows of header in a CSV. if 0, there are only data rows",
        default=1,
        type=int,
    )
    group.add_argument(
        "--column_names",
        help="comma-separated list of names of columns. "
        "used in the absence of a header row or to rename columns",
        default=None,
        type=str,
    )
    group.add_argument(
        "--separator",
        help="field delimiter in text or CSV file",
        default=",",
        type=str,
    )
    group.add_argument(
        "--chunksize",
        help="number of input rows to process in a chunk. recommend using"
        " a smaller chunk size for input with wider rows",
        default=500_000,
        type=int,
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
        "--add_hipscat_index",
        help="Option to generate the _hipscat_index column "
        "a spatially aware index for read and join optimization",
        action="store_true",
    )
    group.add_argument(
        "--overwrite",
        help="if set, any existing catalog data will be overwritten",
        action="store_true",
    )
    group.add_argument(
        "--no_overwrite",
        help="if set, the pipeline will exit if existing output is found",
        dest="overwrite",
        action="store_false",
    )
    group.add_argument(
        "--resume",
        help="if set, the pipeline will try to resume from a previous failed pipeline progress",
        action="store_true",
    )
    group.add_argument(
        "--no_resume",
        help="if set, the pipeline will exit if existing intermediate files are found",
        dest="resume",
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

    return ImportArguments(
        catalog_name=args.catalog_name,
        input_path=args.input_path,
        input_format=args.input_format,
        input_file_list=(
            args.input_file_list.split(",") if args.input_file_list else None
        ),
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        id_column=args.id_column,
        add_hipscat_index=args.add_hipscat_index,
        output_path=args.output_path,
        overwrite=args.overwrite,
        highest_healpix_order=args.highest_healpix_order,
        pixel_threshold=args.pixel_threshold,
        debug_stats_only=args.debug_stats_only,
        file_reader=get_file_reader(
            args.input_format,
            chunksize=args.chunksize,
            header=args.header_rows if args.header_rows != 0 else None,
            schema_file=args.schema_file,
            column_names=(args.column_names.split(",") if args.column_names else None),
            separator=args.separator,
        ),
        tmp_dir=args.tmp_dir,
        progress_bar=args.progress_bar,
        dask_tmp=args.dask_tmp,
        dask_n_workers=args.dask_n_workers,
        dask_threads_per_worker=args.dask_threads_per_worker,
    )
