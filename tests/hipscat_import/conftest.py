import os
import re

import numpy.testing as npt
import pandas as pd
import pytest
from dask.distributed import Client


@pytest.fixture(scope="package", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client()
    yield client
    client.close()


# pylint: disable=missing-function-docstring, redefined-outer-name
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, "data")


@pytest.fixture
def small_sky_dir(test_data_dir):
    return os.path.join(test_data_dir, "small_sky")


@pytest.fixture
def small_sky_single_file(test_data_dir):
    return os.path.join(test_data_dir, "small_sky", "catalog.csv")


@pytest.fixture
def blank_data_dir(test_data_dir):
    return os.path.join(test_data_dir, "blank")


@pytest.fixture
def blank_data_file(test_data_dir):
    return os.path.join(test_data_dir, "blank", "blank.csv")


@pytest.fixture
def empty_data_dir(test_data_dir):
    return os.path.join(test_data_dir, "empty")


@pytest.fixture
def formats_headers_csv(test_data_dir):
    return os.path.join(test_data_dir, "test_formats", "headers.csv")


@pytest.fixture
def formats_pipe_csv(test_data_dir):
    return os.path.join(test_data_dir, "test_formats", "pipe_delimited.csv")


@pytest.fixture
def formats_fits(test_data_dir):
    return os.path.join(test_data_dir, "test_formats", "small_sky.fits")


@pytest.fixture
def small_sky_parts_dir(test_data_dir):
    return os.path.join(test_data_dir, "small_sky_parts")


@pytest.fixture
def small_sky_file0(test_data_dir):
    return os.path.join(test_data_dir, "small_sky_parts", "catalog_00_of_05.csv")


@pytest.fixture
def parquet_shards_dir(test_data_dir):
    return os.path.join(test_data_dir, "parquet_shards")


@pytest.fixture
def parquet_shards_shard_44_0(test_data_dir):
    return os.path.join(
        test_data_dir, "parquet_shards", "dir_0", "pixel_44", "shard_0.parquet"
    )


@pytest.fixture
def mixed_schema_csv_dir(test_data_dir):
    return os.path.join(test_data_dir, "mixed_schema")


@pytest.fixture
def mixed_schema_csv_parquet(test_data_dir):
    return os.path.join(test_data_dir, "mixed_schema", "schema.parquet")


@pytest.fixture
def resume_dir(test_data_dir):
    return os.path.join(test_data_dir, "resume")


@pytest.fixture
def assert_text_file_matches():
    def assert_text_file_matches(expected_lines, file_name):
        """Convenience method to read a text file and compare the contents, line for line.

        When file contents get even a little bit big, it can be difficult to see
        the difference between an actual file and the expected contents without
        increased testing verbosity. This helper compares files line-by-line,
        using the provided strings or regular expressions.

        Notes:
            Because we check strings as regular expressions, you may need to escape some
            contents of `expected_lines`.

        Args:
            expected_lines(:obj:`string array`) list of strings, formatted as regular expressions.
            file_name (str): fully-specified path of the file to read
        """
        assert os.path.exists(file_name), f"file not found [{file_name}]"
        with open(
            file_name,
            "r",
            encoding="utf-8",
        ) as metadata_file:
            contents = metadata_file.readlines()

        assert len(expected_lines) == len(
            contents
        ), f"files not the same length ({len(contents)} vs {len(expected_lines)})"
        for i, expected in enumerate(expected_lines):
            assert re.match(expected, contents[i]), (
                f"files do not match at line {i+1} "
                f"(actual: [{contents[i]}] vs expected: [{expected}])"
            )

    return assert_text_file_matches


@pytest.fixture
def assert_parquet_file_ids():
    def assert_parquet_file_ids(file_name, id_column, expected_ids):
        """
        Convenience method to read a parquet file and compare the object IDs to
        a list of expected objects.

        Args:
            file_name (str): fully-specified path of the file to read
            id_column (str): column in the parquet file to read IDs from
            expected_ids (:obj:`int[]`): list of expected ids in `id_column`
        """
        assert os.path.exists(file_name), f"file not found [{file_name}]"

        data_frame = pd.read_parquet(file_name, engine="pyarrow")
        assert id_column in data_frame.columns
        ids = data_frame[id_column].tolist()

        assert len(ids) == len(
            expected_ids
        ), f"object list not the same size ({len(ids)} vs {len(expected_ids)})"

        npt.assert_array_equal(ids, expected_ids)

    return assert_parquet_file_ids
