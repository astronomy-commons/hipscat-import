"""File reading generators for common file types."""

import abc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

# pylint: disable=too-few-public-methods,too-many-arguments


def get_file_reader(
    file_format,
    chunksize=500_000,
    header="infer",
    schema_file=None,
    column_names=None,
    type_map=None,
    separator=",",
):
    """Get a generator file reader for common file types"""
    if "csv" in file_format:
        return CsvReader(
            chunksize=chunksize,
            header=header,
            schema_file=schema_file,
            column_names=column_names,
            type_map=type_map,
            separator=separator,
        )
    if file_format == "fits":
        return FitsReader()
    if file_format == "parquet":
        return ParquetReader(chunksize=chunksize)

    raise NotImplementedError(f"File Format: {file_format} not supported")


class InputReader(abc.ABC):
    """Base class for chunking file readers."""

    @abc.abstractmethod
    def read(self, input_file):
        """Read the input file, or chunk of the input file.

        Args:
            input_file(str) path to the input file.
        Yields:
            DataFrame containing chunk of file info.
        """

    @abc.abstractmethod
    def provenance_info(self) -> dict:
        """Create dictionary of parameters for provenance tracking.
        Returns:
            dictionary with all argument_name -> argument_value as key -> value pairs.
        """


class CsvReader(InputReader):
    """CSV reader for the most common CSV reading arguments."""

    def __init__(
        self,
        chunksize=500_000,
        header="infer",
        schema_file=None,
        column_names=None,
        type_map=None,
        separator=",",
    ):
        self.chunksize = chunksize
        self.header = header
        self.schema_file = schema_file
        self.column_names = column_names
        self.type_map = type_map
        self.separator = separator

    def read(self, input_file):
        """Read CSV using chunked file reader"""
        if self.schema_file:
            schema_parquet = pd.read_parquet(self.schema_file, use_nullable_dtypes=True)

        use_column_names = None
        if self.column_names:
            use_column_names = self.column_names
        elif not self.header and self.schema_file:
            use_column_names = schema_parquet.columns

        use_type_map = None
        if self.type_map:
            use_type_map = self.type_map
        elif self.schema_file:
            use_type_map = schema_parquet.dtypes.to_dict()

        with pd.read_csv(
            input_file,
            chunksize=self.chunksize,
            sep=self.separator,
            header=self.header,
            names=use_column_names,
            dtype=use_type_map,
        ) as reader:
            for chunk in reader:
                yield chunk

    def provenance_info(self) -> dict:
        str_type_map = {}
        if self.type_map:
            str_type_map = {key: str(value) for (key, value) in self.type_map.items()}
        provenance_info = {
            "input_reader_type": "CsvReader",
            "chunksize": self.chunksize,
            "header": self.header,
            "schema_file": self.schema_file,
            "separator": self.separator,
            "column_names": self.column_names,
            "type_map": str_type_map,
        }
        return provenance_info


class FitsReader(InputReader):
    """Chunked FITS file reader."""

    def __init__(
        self,
        chunksize=500_000,
        column_names=None,
        skip_column_names=None,
    ):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = skip_column_names

    def read(self, input_file):
        """Read chunks of rows in a fits file"""
        table = Table.read(input_file, memmap=True)
        if self.column_names:
            table.keep_columns(self.column_names)
        elif self.skip_column_names:
            table.remove_columns(self.skip_column_names)

        total_rows = len(table)
        read_rows = 0

        while read_rows < total_rows:
            yield table[read_rows : read_rows + self.chunksize].to_pandas()
            read_rows += self.chunksize

    def provenance_info(self) -> dict:
        provenance_info = {
            "input_reader_type": "FitsReader",
            "chunksize": self.chunksize,
            "column_names": self.column_names,
            "skip_column_names": self.skip_column_names,
        }
        return provenance_info


class ParquetReader(InputReader):
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

    def provenance_info(self) -> dict:
        provenance_info = {
            "input_reader_type": "ParquetReader",
            "chunksize": self.chunksize,
        }
        return provenance_info
