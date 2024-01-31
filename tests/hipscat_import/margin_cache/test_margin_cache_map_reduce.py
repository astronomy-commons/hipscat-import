import os

import pandas as pd
import pytest
from hipscat.io import paths
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.margin_cache import margin_cache_map_reduce
from hipscat_import.pipeline_resume_plan import get_pixel_cache_directory

keep_cols = ["weird_ra", "weird_dec"]

drop_cols = [
    "partition_order",
    "partition_pixel",
    "margin_check",
    "margin_pixel",
    "is_trunc",
]


def validate_result_dataframe(df_path, expected_len):
    res_df = pd.read_parquet(df_path)

    assert len(res_df) == expected_len

    cols = res_df.columns.values.tolist()

    for col in keep_cols:
        assert col in cols

    for col in drop_cols:
        assert col not in cols


@pytest.mark.timeout(5)
def test_to_pixel_shard_equator(tmp_path, basic_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        basic_data_shard_df,
        margin_threshold=360.0,
        output_path=tmp_path,
        ra_column="weird_ra",
        dec_column="weird_dec",
    )

    path = os.path.join(tmp_path, "order_1", "dir_0", "pixel_21", "Norder=1", "Dir=0", "Npix=0.parquet")

    assert os.path.exists(path)

    validate_result_dataframe(path, 2)


@pytest.mark.timeout(5)
def test_to_pixel_shard_polar(tmp_path, polar_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        polar_data_shard_df,
        margin_threshold=360.0,
        output_path=tmp_path,
        ra_column="weird_ra",
        dec_column="weird_dec",
    )

    path = os.path.join(tmp_path, "order_2", "dir_0", "pixel_15", "Norder=2", "Dir=0", "Npix=0.parquet")

    assert os.path.exists(path)

    validate_result_dataframe(path, 360)


@pytest.mark.dask
def test_reduce_margin_shards(tmp_path, basic_data_shard_df):
    partition_dir = get_pixel_cache_directory(tmp_path, HealpixPixel(1, 21))
    shard_dir = paths.pixel_directory(partition_dir, 1, 21)

    os.makedirs(shard_dir)

    first_shard_path = paths.pixel_catalog_file(partition_dir, 1, 0)
    second_shard_path = paths.pixel_catalog_file(partition_dir, 1, 1)

    shard_df = basic_data_shard_df.drop(columns=["partition_order", "partition_pixel", "margin_pixel"])

    shard_df.to_parquet(first_shard_path)
    shard_df.to_parquet(second_shard_path)

    margin_cache_map_reduce.reduce_margin_shards(tmp_path, 1, 21)

    result_path = paths.pixel_catalog_file(tmp_path, 1, 21)

    validate_result_dataframe(result_path, 720)
    assert not os.path.exists(shard_dir)
