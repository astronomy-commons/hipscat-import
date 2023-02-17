"""Test dataframe-generating file readers"""
import numpy as np
import pandas as pd

from hipscat_import.file_readers import CsvReader, ParquetReader, fits_reader


def test_csv_reader(small_sky_single_file):
    """Verify we can read the csv file into a single data frame."""
    total_chunks = 0
    for frame in CsvReader().read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1


def test_csv_reader_chunks(small_sky_single_file):
    """Verify we can read the csv file into multiple data frames."""
    total_chunks = 0
    for frame in CsvReader(chunksize=15).read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) <= 15

    assert total_chunks == 9

    total_chunks = 0
    for frame in CsvReader(chunksize=1).read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) == 1

    assert total_chunks == 131


def test_csv_reader_no_headers(small_sky_single_file):
    """Verify we can read the csv file without a header row."""
    total_chunks = 0
    for frame in CsvReader(header=None).read(small_sky_single_file):
        total_chunks += 1
        # The "header" row is interpreted as just another data row
        assert len(frame) == 132

    assert total_chunks == 1


def test_csv_reader_pipe_delimited(formats_pipe_csv):
    """Verify we can read a pipe-delimited csv file without a header row."""
    total_chunks = 0
    for frame in CsvReader(header=None, separator="|").read(formats_pipe_csv):
        total_chunks += 1
        assert len(frame) == 3
        assert np.all(frame[0] == ["AA", "BB", "CC"])

    assert total_chunks == 1

    ## Provide header names and types in a few formats
    frame = next(
        CsvReader(
            header=None,
            separator="|",
            column_names=["letters", "ints", "empty", "numeric"],
        ).read(formats_pipe_csv)
    )
    assert len(frame) == 3
    assert np.all(frame["letters"] == ["AA", "BB", "CC"])
    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "letters": object,
        "ints": int,
        "empty": float,
        "numeric": float,
    }
    assert np.all(column_types == expected_column_types)

    frame = next(
        CsvReader(
            header=None,
            separator="|",
            column_names=["letters", "ints", "empty", "numeric"],
            type_map={
                "letters": object,
                "ints": int,
                "empty": 'Int64',
                "numeric": int,
            },
        ).read(formats_pipe_csv)
    )
    assert len(frame) == 3
    assert np.all(frame["letters"] == ["AA", "BB", "CC"])
    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "letters": object,
        "ints": int,
        "empty": pd.Int64Dtype(),
        "numeric": int,
    }
    assert np.all(column_types == expected_column_types)


def test_parquet_reader(parquet_shards_shard_44_0):
    """Verify we can read the parquet file into a single data frame."""
    total_chunks = 0
    for frame in ParquetReader().read(parquet_shards_shard_44_0):
        total_chunks += 1
        assert len(frame) == 7

    assert total_chunks == 1


def test_parquet_reader_chunked(parquet_shards_shard_44_0):
    """Verify we can read the parquet file into a single data frame."""
    total_chunks = 0
    for frame in ParquetReader(chunksize=1).read(parquet_shards_shard_44_0):
        total_chunks += 1
        assert len(frame) == 1
    assert total_chunks == 7


def test_read_fits(formats_fits):
    """Success case - fits file that exists being read as fits"""
    total_chunks = 0
    for frame in fits_reader(formats_fits):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1
