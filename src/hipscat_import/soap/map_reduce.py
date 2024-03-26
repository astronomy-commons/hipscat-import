"""Inner methods for SOAP"""

from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hipscat.catalog.association_catalog.partition_join_info import PartitionJoinInfo
from hipscat.io import FilePointer, file_io, paths
from hipscat.io.file_io.file_pointer import get_fs, strip_leading_slash_for_pyarrow
from hipscat.io.parquet_metadata import get_healpix_pixel_from_metadata
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from hipscat_import.pipeline_resume_plan import get_pixel_cache_directory
from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.resume_plan import SoapPlan


def _count_joins_for_object(source_data, source_pixel, object_pixel, soap_args):
    object_path = paths.pixel_catalog_file(
        catalog_base_dir=soap_args.object_catalog_dir,
        pixel_order=object_pixel.order,
        pixel_number=object_pixel.pixel,
    )
    object_data = file_io.load_parquet_to_pandas(
        object_path, columns=[soap_args.object_id_column], storage_options=soap_args.object_storage_options
    ).set_index(soap_args.object_id_column)

    joined_data = source_data.merge(object_data, how="inner", left_index=True, right_index=True)

    rows_written = len(joined_data)
    if not soap_args.write_leaf_files or rows_written == 0:
        return rows_written

    # Prepare the dataframe columns
    prepared_data = pd.DataFrame(
        data={
            "object_id": joined_data.index.values,
            "source_id": (
                joined_data.index.values
                if soap_args.source_object_id_column == soap_args.source_id_column
                else joined_data[soap_args.source_id_column]
            ),
            "Norder": np.full(rows_written, fill_value=object_pixel.order, dtype=np.uint8),
            "Dir": np.full(rows_written, fill_value=object_pixel.dir, dtype=np.uint64),
            "Npix": np.full(rows_written, fill_value=object_pixel.pixel, dtype=np.uint64),
            "join_Norder": np.full(rows_written, fill_value=source_pixel.order, dtype=np.uint8),
            "join_Dir": np.full(rows_written, fill_value=source_pixel.dir, dtype=np.uint64),
            "join_Npix": np.full(rows_written, fill_value=source_pixel.pixel, dtype=np.uint64),
        },
    ).drop_duplicates()

    # Write to parquet file.
    pixel_dir = get_pixel_cache_directory(soap_args.tmp_path, object_pixel)
    file_io.make_directory(pixel_dir, exist_ok=True)
    output_file = file_io.append_paths_to_pointer(
        pixel_dir, f"source_{source_pixel.order}_{source_pixel.pixel}.parquet"
    )
    prepared_data.to_parquet(output_file, index=False)

    return rows_written


def _write_count_results(cache_path, source_healpix, results):
    """Build a nice dataframe with pretty columns and rows"""
    num_results = len(results)
    dataframe = pd.DataFrame(results, columns=["Norder", "Npix", "num_rows"])

    dataframe["Dir"] = [int(order / 10_000) * 10_000 if order >= 0 else -1 for order, _, _ in results]
    dataframe["join_Norder"] = np.full(num_results, fill_value=source_healpix.order, dtype=np.uint8)
    dataframe["join_Dir"] = np.full(num_results, fill_value=source_healpix.dir, dtype=np.uint64)
    dataframe["join_Npix"] = np.full(num_results, fill_value=source_healpix.pixel, dtype=np.uint64)

    ## Reorder columns.
    dataframe = dataframe[["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]]

    file_io.write_dataframe_to_csv(
        dataframe=dataframe,
        file_pointer=file_io.append_paths_to_pointer(
            cache_path, f"{source_healpix.order}_{source_healpix.pixel}.csv"
        ),
        index=False,
    )


def count_joins(soap_args: SoapArguments, source_pixel: HealpixPixel, object_pixels: List[HealpixPixel]):
    """Count the number of equijoined sources in the object pixels.
    If any un-joined source pixels remain, stretch out to neighboring object pixels.

    Args:
        soap_args(`hipscat_import.soap.SoapArguments`): set of arguments for pipeline execution
        source_pixel(HealpixPixel): order and pixel for the source catalog single pixel.
        object_pixels(List[HealpixPixel]): set of tuples of order and pixel for the partitions
            of the object catalog to be joined.
    """
    source_path = paths.pixel_catalog_file(
        catalog_base_dir=file_io.get_file_pointer_from_path(soap_args.source_catalog_dir),
        pixel_order=source_pixel.order,
        pixel_number=source_pixel.pixel,
    )
    if soap_args.write_leaf_files and soap_args.source_object_id_column != soap_args.source_id_column:
        read_columns = [soap_args.source_object_id_column, soap_args.source_id_column]
    else:
        read_columns = [soap_args.source_object_id_column]
    source_data = file_io.load_parquet_to_pandas(
        source_path, columns=read_columns, storage_options=soap_args.source_storage_options
    ).set_index(soap_args.source_object_id_column)

    remaining_sources = len(source_data)
    results = []

    for object_pixel in object_pixels:
        if remaining_sources < 1:
            break
        join_count = _count_joins_for_object(
            source_data,
            source_pixel,
            object_pixel,
            soap_args,
        )
        results.append([object_pixel.order, object_pixel.pixel, join_count])
        remaining_sources -= join_count

    ## mark that some sources were not joined
    if remaining_sources > 0:
        results.append([-1, -1, remaining_sources])

    _write_count_results(soap_args.tmp_path, source_pixel, results)


def combine_partial_results(input_path, output_path, output_storage_options) -> int:
    """Combine many partial CSVs into single partition join info.
    Also write out a debug file with counts of unmatched sources, if any.

    Args:
        input_path(str): intermediate directory with partial result CSVs. likely, the
            directory used in the previous `count_joins` call as `cache_path`
        output_path(str): directory to write the combined results CSVs.

    Returns:
        integer that is the sum of all matched num_rows.
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
        storage_options=output_storage_options,
    )

    if len(unmatched) > 0:
        file_io.write_dataframe_to_csv(
            dataframe=unmatched,
            file_pointer=file_io.append_paths_to_pointer(output_path, "unmatched_sources.csv"),
            index=False,
            storage_options=output_storage_options,
        )

    primary_only = matched.groupby(["Norder", "Dir", "Npix"])["num_rows"].sum().reset_index()
    file_io.write_dataframe_to_csv(
        dataframe=primary_only,
        file_pointer=file_io.append_paths_to_pointer(output_path, "partition_info.csv"),
        index=False,
        storage_options=output_storage_options,
    )

    join_info = PartitionJoinInfo(matched)
    join_info.write_to_metadata_files(output_path, storage_options=output_storage_options)

    return primary_only["num_rows"].sum()


def reduce_joins(
    soap_args: SoapArguments, object_pixel: HealpixPixel, object_key: str, delete_input_files: bool = True
):
    """Reduce join tables into one parquet file per object-pixel, with one row-group
    inside per source pixel."""
    pixel_dir = get_pixel_cache_directory(soap_args.tmp_path, object_pixel)
    # If there's no directory, this implies there were no matches to this object pixel
    # earlier in the pipeline. Move on.
    if not file_io.does_file_or_directory_exist(pixel_dir):
        return
    # Find all of the constituent files / source pixels. Create a list of PyArrow Tables from those
    # parquet files. We need to know the schema before we create the ParquetWriter.
    shard_file_list = file_io.find_files_matching_path(pixel_dir, "source**.parquet")

    if len(shard_file_list) == 0:
        return

    ## We want to order the row groups in a "breadth-first" sorting. Determine our sorting
    ## via the metadata, then read the tables in using that sorting.
    healpix_pixels = []
    for shard_file_name in shard_file_list:
        healpix_pixels.append(
            get_healpix_pixel_from_metadata(pq.read_metadata(shard_file_name), "join_Norder", "join_Npix")
        )

    argsort = get_pixel_argsort(healpix_pixels)
    shard_file_list = np.array(shard_file_list)[argsort]

    shards = []
    for shard_file_name in shard_file_list:
        shards.append(pq.read_table(shard_file_name))

    # Write all of the shards into a single parquet file, one row-group-per-shard.
    starting_catalog_path = FilePointer(str(soap_args.catalog_path))
    destination_dir = paths.pixel_directory(starting_catalog_path, object_pixel.order, object_pixel.pixel)
    file_io.make_directory(destination_dir, exist_ok=True, storage_options=soap_args.output_storage_options)

    output_file = paths.pixel_catalog_file(starting_catalog_path, object_pixel.order, object_pixel.pixel)
    file_system, output_file = get_fs(
        file_pointer=output_file, storage_options=soap_args.output_storage_options
    )
    output_file = strip_leading_slash_for_pyarrow(output_file, protocol=file_system.protocol)
    with pq.ParquetWriter(output_file, shards[0].schema, filesystem=file_system) as writer:
        for table in shards:
            writer.write_table(table)

    # Delete the intermediate shards.
    if delete_input_files:
        file_io.remove_directory(pixel_dir, ignore_errors=True)

    SoapPlan.reducing_key_done(tmp_path=soap_args.tmp_path, reducing_key=object_key)
