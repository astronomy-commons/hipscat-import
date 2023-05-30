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
    skip_column_names=None,
    type_map=None,
    separator=",",
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
        header (int, list of int, None, default 'infer'): for CSV files, rows to
            use as the header with column names
        schema_file (str): path to a parquet schema file. if provided, header names
            and column types will be pulled from the parquet schema metadata.
        column_names (list[str]): for CSV files, the names of columns if no header
            is available. for fits files, a list of columns to *keep*.
        skip_column_names (list[str]): for fits files, a list of columns to remove.
        type_map (dict): for CSV files, the data types to use for columns
        separator (str): for CSV files, the character used for separation.
    """
    if "csv" in file_format:
        return CsvReader(
            chunksize=chunksize,
            header=header,
            schema_file=schema_file,
            column_names=column_names,
            type_map=type_map,
            separator=separator,
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
    def read(self, input_file):
        """Read the input file, or chunk of the input file.

        Args:
            input_file(str): path to the input file.
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
        separator (str): the character used for separation.
    """

    def __init__(
        self,
        chunksize=500_000,
        header="infer",
        schema_file=None,
        column_names=None,
        type_map=None,
        separator=",",
        **kwargs,
    ):
        self.chunksize = chunksize
        self.header = header
        self.schema_file = schema_file
        self.column_names = column_names
        self.type_map = type_map
        self.separator = separator
        self.kwargs = kwargs

    def read(self, input_file):
        if self.schema_file:
            schema_parquet = pd.read_parquet(
                self.schema_file, dtype_backend="numpy_nullable"
            )

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
            **self.kwargs,
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
    """Chunked FITS file reader.

    There are two column-level arguments for reading fits files:
    `column_names` and `skip_column_names`.

        - If neither is provided, we will read and process all columns in the fits file.
        - If `column_names` is given, we will use *only* those names, and
          `skip_column_names` will be ignored.
        - If `skip_column_names` is provided, we will remove those columns from processing stages.

    NB:
        Uses astropy table memmap to avoid reading the entire file into memory.

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

    def __init__(
        self, chunksize=500_000, column_names=None, skip_column_names=None, **kwargs
    ):
        self.chunksize = chunksize
        self.column_names = column_names
        self.skip_column_names = skip_column_names
        self.kwargs = kwargs

    def read(self, input_file):
        table = Table.read(input_file, memmap=True, **self.kwargs)
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

    def read(self, input_file):
        parquet_file = pq.read_table(input_file, **self.kwargs)
        for smaller_table in parquet_file.to_batches(max_chunksize=self.chunksize):
            yield pa.Table.from_batches([smaller_table]).to_pandas()

    def provenance_info(self) -> dict:
        provenance_info = {
            "input_reader_type": "ParquetReader",
            "chunksize": self.chunksize,
        }
        return provenance_info
