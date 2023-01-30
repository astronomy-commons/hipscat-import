"""Import a set of non-hipscat files using dask for parallelization"""

import os

import dask.dataframe as dd
import healpy as hp
import hipscat.io.paths as paths
import hipscat.pixel_math as pixel_math
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

HIPSCAT_PIXEL_COLUMN = "hipscat_pixel"


def _write_shard_file(pixel_data, cache_path, shard_suffix):
    # Get pixel number from column, check they're all the same, and remove the column
    pixel_number = pixel_data[HIPSCAT_PIXEL_COLUMN].iloc[0]
    assert (pixel_data[HIPSCAT_PIXEL_COLUMN] == pixel_number).all()
    pixel_data = pixel_data.drop(columns=[HIPSCAT_PIXEL_COLUMN])

    pixel_dir = os.path.join(cache_path, f"pixel_{pixel_number}")
    os.makedirs(pixel_dir, exist_ok=True)
    output_file = os.path.join(pixel_dir, f"shard_{shard_suffix}.parquet")
    pixel_data.to_parquet(output_file)


def map_to_pixels(
    input_file,
    file_format,
    shard_suffix,
    highest_order,
    ra_column,
    dec_column,
    cache_path=None,
    filter_function=None,
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
        data = dd.read_csv(input_file)

    elif file_format == "fits":
        data = dd.from_pandas(
            Table.read(input_file, format="fits").to_pandas(), chunksize=500_000
        )
    elif file_format == "parquet":
        data = dd.read_parquet(input_file, engine="pyarrow")
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
    data[HIPSCAT_PIXEL_COLUMN] = hp.ang2pix(
        2**highest_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )

    data = data.compute()

    # Output a shard file per pixel
    if cache_path:
        data.groupby([HIPSCAT_PIXEL_COLUMN]).apply(
            _write_shard_file, cache_path, shard_suffix
        )

    # Populate our histogram
    counts = data.hipscat_pixel.value_counts()
    counts.columns = [HIPSCAT_PIXEL_COLUMN, "counts"]
    for index, count in counts.items():
        histo[index] = count

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

    input_directories = []

    for pixel in origin_pixel_numbers:
        pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
        input_directories.append(pixel_dir)

    tables = []
    for path in input_directories:
        tables.append(pq.read_table(path))
    merged_table = pa.concat_tables(tables)
    if id_column:
        merged_table = merged_table.sort_by(id_column)

    pq.write_table(merged_table, where=destination_file)

    rows_written = len(merged_table)

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )
