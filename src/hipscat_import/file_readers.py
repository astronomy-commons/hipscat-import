"""File reading generators for common file types."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table


def get_file_reader(file_format):
    """Get a generator file reader for common file types"""
    if "csv" in file_format:
        return csv_reader
    if file_format == "fits":
        return fits_reader
    if file_format == "parquet":
        return parquet_reader

    raise NotImplementedError(f"File Format: {file_format} not supported")


def csv_reader(input_file):
    """Read chunks of 500k rows in a CSV"""
    with pd.read_csv(input_file, chunksize=500_000) as reader:
        for chunk in reader:
            yield chunk


def fits_reader(input_file):
    """Read the whole fits file and return"""
    yield Table.read(input_file, format="fits").to_pandas()


def parquet_reader(input_file):
    """Read chunks of 500k rows in a parquet file."""
    parquet_file = pq.read_table(input_file)
    for smaller_table in parquet_file.to_batches(max_chunksize=500_000):
        yield pa.Table.from_batches([smaller_table]).to_pandas()
