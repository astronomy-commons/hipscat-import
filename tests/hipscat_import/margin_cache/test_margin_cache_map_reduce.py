import pandas as pd
import pytest
from hipscat.io import file_io, paths

from hipscat_import.margin_cache import margin_cache_map_reduce

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
        margin_threshold=0.1,
        output_path=tmp_path,
        margin_order=3,
        ra_column="weird_ra",
        dec_column="weird_dec",
    )

    path = file_io.append_paths_to_pointer(
        tmp_path, "Norder=1/Dir=0/Npix=21/Norder=1/Dir=0/Npix=0.parquet"
    )

    assert file_io.does_file_or_directory_exist(path)

    validate_result_dataframe(path, 46)

@pytest.mark.timeout(5)
def test_to_pixel_shard_polar(tmp_path, polar_data_shard_df):
    margin_cache_map_reduce._to_pixel_shard(
        polar_data_shard_df,
        margin_threshold=0.1,
        output_path=tmp_path,
        margin_order=3,
        ra_column="weird_ra",
        dec_column="weird_dec",
    )

    path = file_io.append_paths_to_pointer(
        tmp_path, "Norder=2/Dir=0/Npix=15/Norder=2/Dir=0/Npix=0.parquet"
    )

    assert file_io.does_file_or_directory_exist(path)

    validate_result_dataframe(path, 317)

def test_reduce_margin_shards(tmp_path, basic_data_shard_df):
    partition_dir = margin_cache_map_reduce._get_partition_directory(
        tmp_path, 1, 21
    )
    shard_dir = paths.pixel_directory(
        partition_dir, 1, 21
    )

    file_io.make_directory(shard_dir, exist_ok=True)

    first_shard_path = paths.pixel_catalog_file(
        partition_dir, 1, 0
    )
    second_shard_path = paths.pixel_catalog_file(
        partition_dir, 1, 1
    )

    print(first_shard_path)

    shard_df = basic_data_shard_df.drop(columns=[
        "partition_order", 
        "partition_pixel",
        "margin_pixel"
    ])

    shard_df.to_parquet(first_shard_path)
    shard_df.to_parquet(second_shard_path)

    margin_cache_map_reduce.reduce_margin_shards(tmp_path, 1, 21)

    result_path = paths.pixel_catalog_file(tmp_path, 1, 21)

    validate_result_dataframe(result_path, 720)
