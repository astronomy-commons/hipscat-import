"""Run pass/fail checks and generate verification report of existing hipscat table."""

import logging

import numpy as np
import numpy.testing as npt
from hipscat.catalog.partition_info import PartitionInfo
from hipscat.io import file_io, paths
from hipscat.pixel_math.healpix_pixel import INVALID_PIXEL
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from hipscat_import.verification.arguments import VerificationArguments

# pylint: disable=logging-fstring-interpolation


logger = logging.getLogger("hipscat_verification")


def run(args):
    """Run verification pipeline."""
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    ## Set up our logger that we'll use for all checks ------------------------
    logging.basicConfig(filename=args.output_path, level=logging.INFO)
    logger.info("Started")

    ## Compare the pixels in _metadata with partition_info.csv ----------------
    metadata_file = paths.get_parquet_metadata_pointer(args.input_catalog_path)
    partition_info_file = paths.get_partition_info_pointer(args.input_catalog_path)

    expected_pixels = PartitionInfo.read_from_file(
        metadata_file, storage_options=args.input_storage_options
    ).get_healpix_pixels()

    logger.info(f"Expecting {len(expected_pixels)} count of pixels")

    if file_io.does_file_or_directory_exist(partition_info_file, storage_options=args.input_storage_options):
        csv_pixels = PartitionInfo.read_from_csv(partition_info_file).get_healpix_pixels()
        npt.assert_array_equal(
            expected_pixels,
            csv_pixels,
            err_msg="Partition pixels differ between _metadata and partition_info.csv files",
            strict=True,
        )
        logger.info(f"Found {len(csv_pixels)} in partition_info.csv file")
    else:
        logger.warning(
            "Catalog doesn't contain a partition_info.csv file. "
            "This is not required, but is strongly recommended."
        )

    ## Load as parquet dataset. Allow errors, and check pixel set against _metadata
    ignore_prefixes = [
        "_common_metadata",
        "_metadata",
        "catalog_info.json",
        "provenance_info.json",
        "partition_info.csv",
        "point_map.fits",
    ]

    (dataset_path, dataset) = file_io.read_parquet_dataset(
        args.input_catalog_path,
        storage_options=args.input_storage_options,
        ignore_prefixes=ignore_prefixes,
        exclude_invalid_files=False,
    )

    parquet_path_pixels = []
    for hips_file in dataset.files:
        relative_path = hips_file[len(dataset_path) :]

        healpix_pixel = paths.get_healpix_from_path(relative_path)
        if healpix_pixel == INVALID_PIXEL:
            message = f"Could not derive partition pixel from parquet path: {relative_path}"
            if args.fail_fast:
                raise ValueError(message)
            logger.error(message)

        parquet_path_pixels.append(healpix_pixel)

    argsort = get_pixel_argsort(parquet_path_pixels)
    parquet_path_pixels = np.array(parquet_path_pixels)[argsort]

    npt.assert_array_equal(
        expected_pixels,
        parquet_path_pixels,
        err_msg="Partition pixels differ between _metadata and parquet paths",
        strict=True,
    )

    logger.info(f"Found {len(parquet_path_pixels)} in parquet paths")

    logger.info("Finished")
