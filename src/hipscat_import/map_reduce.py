"""Import a set of non-hipscat files using dask for parallelization"""

import os

import healpy as hp
import numpy as np
import pandas as pd
from astropy.table import Table
from hipscat import pixel_math
from hipscat.io import paths

# pylint: disable=too-many-locals,too-many-arguments

def map_to_pixels(
    input_file,
    file_format,
    shard_suffix,
    highest_order,
    ra_column,
    dec_column,
    cache_path=None,
    filter_function=None,
    schema_file=None,
):
    """Map a file of input objects to their healpix pixels."""

    # Perform checks on the provided path
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found at path: {input_file}")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Directory found at path - requires regular file: {input_file}"
        )

    required_columns = [ra_column, dec_column]

    # Load file using appropriate mechanism
    if "csv" in file_format:
        if schema_file:
            schema_parquet = pd.read_parquet(schema_file)
            data = pd.read_csv(input_file, dtype=schema_parquet.dtypes.to_dict())
        else:
            data = pd.read_csv(input_file)

    elif file_format == "fits":
        data = Table.read(input_file, format="fits").to_pandas()
    elif file_format == "parquet":
        data = pd.read_parquet(input_file, engine="pyarrow")
        data.reset_index(inplace=True)
    else:
        raise NotImplementedError(f"File Format: {file_format} not supported")

    if not all(x in data.columns for x in required_columns):
        raise ValueError(
            f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
        )
    histo = pixel_math.empty_histogram(highest_order)

    # Set up the data we want (filter and find pixel)
    if filter_function:
        data = filter_function(data)
        data.reset_index(inplace=True)
    mapped_pixels = hp.ang2pix(
        2**highest_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )
    mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
    histo[mapped_pixel] += count_at_pixel.astype(np.ulonglong)

    if cache_path:
        for pixel in mapped_pixel:
            data_indexes = np.where(mapped_pixels == pixel)
            filtered_data = data.filter(items=data_indexes[0].tolist(), axis=0)

            pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
            os.makedirs(pixel_dir, exist_ok=True)
            output_file = os.path.join(pixel_dir, f"shard_{shard_suffix}.parquet")
            filtered_data.to_parquet(output_file)
        del filtered_data, data_indexes

    ## Pesky memory!
    del data, mapped_pixels, mapped_pixel, count_at_pixel
    return histo


def reduce_pixel_shards(
    cache_path,
    origin_pixel_numbers,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    id_column,
):
    """Reduce sharded source pixels into destination pixels."""
    destination_dir = paths.pixel_directory(
        output_path, destination_pixel_order, destination_pixel_number
    )
    os.makedirs(destination_dir, exist_ok=True)

    destination_file = paths.pixel_catalog_file(
        output_path, destination_pixel_order, destination_pixel_number
    )

    tables = []
    for pixel in origin_pixel_numbers:
        pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")

        for file in os.listdir(pixel_dir):
            tables.append(
                pd.read_parquet(os.path.join(pixel_dir, file), engine="pyarrow")
            )

    merged_table = pd.concat(tables, ignore_index=True, sort=False)
    if id_column:
        merged_table = merged_table.sort_values(by=id_column)

    rows_written = len(merged_table)

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )

    merged_table.to_parquet(destination_file)

    del merged_table, tables
