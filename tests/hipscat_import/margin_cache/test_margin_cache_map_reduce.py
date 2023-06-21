import healpy as hp
import numpy as np
import pandas as pd
import pytest
from hipscat.io import file_io

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
def test_to_pixel_shard_equator(tmp_path):
    ras = np.arange(0.0, 360.0)
    dec = np.full(360, 0.0)
    ppix = np.full(360, 21)
    porder = np.full(360, 1)
    norder = np.full(360, 1)
    npix = np.full(360, 0)

    test_df = pd.DataFrame(
        data=zip(ras, dec, ppix, porder, norder, npix),
        columns=[
            "weird_ra",
            "weird_dec",
            "partition_pixel",
            "partition_order",
            "Norder",
            "Npix",
        ],
    )

    test_df["margin_pixel"] = hp.ang2pix(
        2**3,
        test_df["weird_ra"].values,
        test_df["weird_dec"].values,
        lonlat=True,
        nest=True,
    )

    margin_cache_map_reduce._to_pixel_shard(
        test_df,
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
def test_to_pixel_shard_polar(tmp_path):
    ras = np.arange(0.0, 360.0)
    dec = np.full(360, 89.9)
    ppix = np.full(360, 15)
    porder = np.full(360, 2)
    norder = np.full(360, 2)
    npix = np.full(360, 0)

    test_df = pd.DataFrame(
        data=zip(ras, dec, ppix, porder, norder, npix),
        columns=[
            "weird_ra",
            "weird_dec",
            "partition_pixel",
            "partition_order",
            "Norder",
            "Npix",
        ],
    )

    test_df["margin_pixel"] = hp.ang2pix(
        2**3,
        test_df["weird_ra"].values,
        test_df["weird_dec"].values,
        lonlat=True,
        nest=True,
    )

    margin_cache_map_reduce._to_pixel_shard(
        test_df,
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
