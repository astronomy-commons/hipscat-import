"""File reading generators for common file types."""

import abc
from typing import Any, Dict, Union

import pyarrow.parquet as pq
from astropy.table import Table
from hipscat.io import FilePointer, file_io

# pylint: disable=too-few-public-methods,too-many-arguments


def get_file_reader(
    file_format,
    chunksize=500_000,
    schema_file=None,
    column_names=None,
    skip_column_names=None,
    type_map=None,
    **kwargs,
):
    """Get a generator file reader for common file types

    Args:
        file_format (str): specifier for the file type and extension.
            Currently supported formats include:

            - `csv`, comma separated values. may also be tab- or pipe-delimited
              includes `.csv.gz` and other compressed csv files
            - `fits`, flexible image transport system. often used for astropy tables.
            - `parquet`, compressed columnar data format

        chunksize (int): number of rows to read in a single iteration.
        schema_file (str): path to a parquet schema file. if provided, header names
            and column types will be pulled from the parquet schema metadata.
        column_names (list[str]): for CSV files, the names of columns if no header
            is available. for fits files, a list of columns to *keep*.
        skip_column_names (list[str]): for fits files, a list of columns to remove.
        type_map (dict): for CSV files, the data types to use for columns
    """
    if "csv" in file_format:
        return CsvReader(
            chunksize=chunksize,
            schema_file=schema_file,
            column_names=column_names,
            type_map=type_map,
            **kwargs,
        )
    if file_format == "fits":
        return FitsReader(
            chunksize=chunksize,
            column_names=column_names,
            skip_column_names=skip_column_names,
            **kwargs,
        )
    if file_format == "parquet":
        return ParquetReader(chunksize=chunksize, **kwargs)

    raise NotImplementedError(f"File Format: {file_format} not supported")


class InputReader(abc.ABC):
    """Base class for chunking file readers."""

    @abc.abstractmethod
    def read(self, input_file, read_columns=None):
        """Read the input file, or chunk of the input file.

        Args:
            input_file(str): path to the input file.
            read_columns(List[str]): subset of columns to read.
                if None, all columns are read
        Yields:
            DataFrame containing chunk of file info.
        """

    @abc.abstractmethod
    def provenance_info(self) -> dict:
        """Create dictionary of parameters for provenance tracking.

        Returns:
            dictionary with all argument_name -> argument_value as key -> value pairs.
        """

    def regular_file_exists(self, input_file, storage_options: Union[Dict[Any, Any], None] = None, **_kwargs):
        """Check that the `input_file` points to a single regular file

        Raises
            FileNotFoundError: if nothing exists at path, or directory found.
        """
        if not file_io.does_file_or_directory_exist(input_file, storage_options=storage_options):
            raise FileNotFoundError(f"File not found at path: {input_file}")
        if not file_io.is_regular_file(input_file, storage_options=storage_options):
            raise FileNotFoundError(f"Directory found at path - requires regular file: {input_file}")


class CsvReader(InputReader):
    """CSV reader for the most common CSV reading arguments.

    This uses `pandas.read_csv`, and you can find more information on
    additional arguments in the pandas documentation:
    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    Attributes:
        chunksize (int): number of rows to read in a single iteration.
        header (int, list of int, None, default 'infer'): rows to
            use as the header with column names
        schema_file (str): path to a parquet schema file. if provided, header names
            and column types will be pulled from the parquet schema metadata.
        column_names (list[str]): the names of columns if no header is available
        type_map (dict): the data types to use for columns
        parquet_kwargs (dict): additional keyword arguments to use when
            reading the parquet schema metadata.
        kwargs (dict): additional keyword arguments to use when reading
            the CSV files.
    """

    def __init__(
        self,
        chunksize=500_000,
        header="infer",
        schema_file=None,
        column_names=None,
        type_map=None,
        parquet_kwargs=None,
        **kwargs,
    ):
        self.chunksize = chunksize
        self.header = header
        self.schema_file = schema_file
        self.column_names = column_names
        self.type_map = type_map
        self.parquet_kwargs = parquet_kwargs
        self.kwargs = kwargs

        if self.schema_file:
            if self.parquet_kwargs is None:
                self.parquet_kwargs = {}
            schema_parquet = file_io.load_parquet_to_pandas(
                FilePointer(self.schema_file),
                **self.parquet_kwargs,
            )

        if self.column_names:
            self.kwargs["names"] = self.column_names
        elif not self.header and self.schema_file:
            self.kwargs["names"] = schema_parquet.columns

        if self.type_map:
            self.kwargs["dtype"] = self.type_map
        elif self.schema_file:
            self.kwargs["dtype"] = schema_parquet.dtypes.to_dict()

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)

        if read_columns:
            self.kwargs["usecols"] = read_columns

        with file_io.load_csv_to_pandas(
            FilePointer(input_file),
            chunksize=self.chunksize,
            header=self.header,
            **self.kwargs,
        ) as reader:
            yield from reader

    def provenance_info(self) -> dict:
        str_kwargs = {}
        if self.type_map:
            str_kwargs = {key: str(value) for (key, value) in self.kwargs.items()}
        provenance_info = {
            "input_reader_type": "CsvReader",
            "chunksize": self.chunksize,
            "schema_file": self.schema_file,
            "column_names": self.column_names,
            "parquet_kwargs": self.parquet_kwargs,
            "kwargs": str_kwargs,
        }
        return provenance_info


class FitsReader(InputReader):
    """Chunked FITS file reader.

    There are two column-level arguments for reading fits files:
    `column_names` and `skip_column_names`.

        - If neither is provided, we will read and process all columns in the fits file.
        - If `column_names` is given, we will use *only* those names, and
          `skip_column_names` will be ignored.
        - If `skip_column_names` is provided, we will remove those columns from processing stages.

    NB: Uses astropy table memmap to avoid reading the entire file into memory.
    See: https://docs.astropy.org/en/stable/io/fits/index.html#working-with-large-files


    Attributes:
        chunksize (int): number of rows of the file to process at once.
            For large files, this can prevent loading the entire file
            into memory at once.
        column_names (list[str]): list of column names to keep. only use
            one of `column_names` or `skip_column_names`
        skip_column_names (list[str]): list of column names to skip. only use
            one of `column_names` or `skip_column_names`
    """

    def __init__(self, chunksize=500_000, column_names=None, skip_column_names=None, **kwargs):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = skip_column_names
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        table = Table.read(input_file, memmap=True, **self.kwargs)
        if read_columns:
            table.keep_columns(read_columns)
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
    """Parquet reader for the most common Parquet reading arguments.

    Attributes:
        chunksize (int): number of rows of the file to process at once.
            For large files, this can prevent loading the entire file
            into memory at once.
    """

    def __init__(self, chunksize=500_000, **kwargs):
        self.chunksize = chunksize
        self.kwargs = kwargs

    def read(self, input_file, read_columns=None):
        self.regular_file_exists(input_file, **self.kwargs)
        parquet_file = pq.ParquetFile(input_file, **self.kwargs)
        for smaller_table in parquet_file.iter_batches(batch_size=self.chunksize, use_pandas_metadata=True):
            yield smaller_table.to_pandas()

    def provenance_info(self) -> dict:
        provenance_info = {
            "input_reader_type": "ParquetReader",
            "chunksize": self.chunksize,
        }
        return provenance_info
