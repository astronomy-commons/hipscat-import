"""Inner methods for SOAP"""

from typing import List

import numpy as np
import pandas as pd
from hipscat.io import file_io
from hipscat.io.paths import pixel_catalog_file
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.soap.arguments import SoapArguments


def _count_joins_for_object(source_data, object_catalog_dir, object_id_column, object_pixel):
    object_path = pixel_catalog_file(
        catalog_base_dir=object_catalog_dir,
        pixel_order=object_pixel.order,
        pixel_number=object_pixel.pixel,
    )
    object_data = file_io.load_parquet_to_pandas(object_path, columns=[object_id_column]).set_index(
        object_id_column
    )

    joined_data = source_data.merge(object_data, how="inner", left_index=True, right_index=True)

    return len(joined_data)


def _write_count_results(cache_path, source_healpix, results):
    """Build a nice dataframe with pretty columns and rows"""
    num_results = len(results)
    dataframe = pd.DataFrame(results, columns=["Norder", "Npix", "num_rows"])

    dataframe["Dir"] = [int(order / 10_000) * 10_000 if order >= 0 else -1 for order, _, _ in results]
    dataframe["join_Norder"] = np.full(num_results, fill_value=source_healpix.order, dtype=np.int32)
    dataframe["join_Dir"] = [int(order / 10_000) * 10_000 for order in dataframe["join_Norder"]]
    dataframe["join_Npix"] = np.full(num_results, fill_value=source_healpix.pixel, dtype=np.int32)

    ## Reorder columns.
    dataframe = dataframe[["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]]

    file_io.write_dataframe_to_csv(
        dataframe=dataframe,
        file_pointer=file_io.append_paths_to_pointer(
            cache_path, f"{source_healpix.order}_{source_healpix.pixel}.csv"
        ),
        index=False,
    )


def count_joins(
    soap_args: SoapArguments, source_pixel: HealpixPixel, object_pixels: List[HealpixPixel], cache_path: str
):
    """Count the number of equijoined sources in the object pixels.
    If any un-joined source pixels remain, stretch out to neighboring object pixels.

    Args:
        soap_args(SoapArguments): set of arguments for pipeline execution
        source_pixel(HealpixPixel): order and pixel for the source catalog single pixel.
        object_pixels(List[HealpixPixel]): set of tuples of order and pixel for the partitions
            of the object catalog to be joined.
        cache_path(str): path to write intermediate results CSV to.
    """
    source_path = pixel_catalog_file(
        catalog_base_dir=file_io.get_file_pointer_from_path(soap_args.source_catalog_dir),
        pixel_order=source_pixel.order,
        pixel_number=source_pixel.pixel,
    )
    source_data = file_io.load_parquet_to_pandas(
        source_path, columns=[soap_args.source_object_id_column]
    ).set_index(soap_args.source_object_id_column)

    remaining_sources = len(source_data)
    results = []

    for object_pixel in object_pixels:
        if remaining_sources < 1:
            break
        join_count = _count_joins_for_object(
            source_data,
            soap_args.object_catalog_dir,
            soap_args.object_id_column,
            object_pixel,
        )
        results.append([object_pixel.order, object_pixel.pixel, join_count])
        remaining_sources -= join_count

    ## mark that some sources were not joined
    if remaining_sources > 0:
        results.append([-1, -1, remaining_sources])

    _write_count_results(cache_path, source_pixel, results)


def combine_partial_results(input_path, output_path):
    """Combine many partial CSVs into single partition join info.
    Also write out a debug file with counts of unmatched sources, if any.

    Args:
        input_path(str): intermediate directory with partial result CSVs. likely, the
            directory used in the previous `count_joins` call as `cache_path`
        output_path(str): directory to write the combined results CSVs.
    """
    partial_files = file_io.find_files_matching_path(input_path, "**.csv")
    partials = []

    for partial_file in partial_files:
        partials.append(file_io.load_csv_to_pandas(partial_file))

    dataframe = pd.concat(partials)

    matched = dataframe.loc[dataframe["Norder"] != -1]
    matched = matched.loc[matched["num_rows"] > 0]
    unmatched = dataframe.loc[dataframe["Norder"] == -1]

    file_io.write_dataframe_to_csv(
        dataframe=matched,
        file_pointer=file_io.append_paths_to_pointer(output_path, "partition_join_info.csv"),
        index=False,
    )

    if len(unmatched) > 0:
        file_io.write_dataframe_to_csv(
            dataframe=unmatched,
            file_pointer=file_io.append_paths_to_pointer(output_path, "unmatched_sources.csv"),
            index=False,
        )
