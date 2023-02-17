"""File reading generators for common file types."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

# pylint: disable=too-few-public-methods,too-many-arguments


def get_file_reader(file_format):
    """Get a generator file reader for common file types"""
    if "csv" in file_format:
        return CsvReader().read
    if file_format == "fits":
        return fits_reader
    if file_format == "parquet":
        return ParquetReader().read

    raise NotImplementedError(f"File Format: {file_format} not supported")


class CsvReader:
    """CSV reader for the most common CSV reading arguments."""

    def __init__(
        self,
        chunksize=500_000,
        header="infer",
        column_names=None,
        type_map=None,
        separator=",",
    ):
        self.chunksize = chunksize
        self.header = header
        self.column_names = column_names
        self.type_map = type_map
        self.separator = separator

    def read(self, input_file):
        """Read CSV using chunked file reader"""
        with pd.read_csv(
            input_file,
            chunksize=self.chunksize,
            sep=self.separator,
            header=self.header,
            names=self.column_names,
            dtype=self.type_map,
        ) as reader:
            for chunk in reader:
                yield chunk


def fits_reader(input_file):
    """Read the whole fits file and return"""
    yield Table.read(input_file, format="fits").to_pandas()


class ParquetReader:
    """Parquet reader for the most common Parquet reading arguments."""

    def __init__(
        self,
        chunksize=500_000,
    ):
        self.chunksize = chunksize

    def read(self, input_file):
        """Read chunks of rows in a parquet file."""
        parquet_file = pq.read_table(input_file)
        for smaller_table in parquet_file.to_batches(max_chunksize=self.chunksize):
            yield pa.Table.from_batches([smaller_table]).to_pandas()
