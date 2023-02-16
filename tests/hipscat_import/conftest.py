import os

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
def mixed_schema_csv_dir(test_data_dir):
    return os.path.join(test_data_dir, "mixed_schema")


@pytest.fixture
def mixed_schema_csv_parquet(test_data_dir):
    return os.path.join(test_data_dir, "mixed_schema", "schema.parquet")
