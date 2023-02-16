"""File reading generators for common file types."""

import pandas as pd
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
    """Read chunks of 1million rows in a CSV"""
    with pd.read_csv(input_file, chunksize=1_000_000) as reader:
        for chunk in reader:
            yield chunk


def fits_reader(input_file):
    """Read the whole fits file and return"""
    yield Table.read(input_file, format="fits").to_pandas()


def parquet_reader(input_file):
    """This should read smaller row groups of a parquet file."""
    yield pd.read_parquet(input_file, engine="pyarrow")
