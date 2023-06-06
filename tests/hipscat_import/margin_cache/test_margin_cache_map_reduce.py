import healpy as hp
import numpy as np
import pandas as pd
import pytest
from hipscat.io import file_io

from hipscat_import.margin_cache import margin_cache_map_reduce


@pytest.mark.timeout(5)
def test_to_pixel_shard(tmp_path):
    ras = np.arange(0.,360.)
    dec = np.full(360, 89.9)
    ppix = np.full(360, 15)
    porder = np.full(360, 2)
    norder = np.full(360, 3)
    npix = np.full(360, 0)

    test_df = pd.DataFrame(
        data=zip(ras, dec, ppix, porder, norder, npix),
        columns=[
            "ra", 
            "dec",
            "partition_pixel",
            "partition_order",
            "Norder",
            "Npix"
        ]
    )

    test_df["margin_pixel"] = hp.ang2pix(
        2**3,
        test_df["ra"].values,
        test_df["dec"].values,
        lonlat=True,
        nest=True
    )

    margin_cache_map_reduce._to_pixel_shard(
        test_df,
        margin_threshold=0.1,
        output_path=tmp_path,
        margin_order=3,
        ra_column="ra",
        dec_column="dec"
    )

    path = file_io.append_paths_to_pointer(
        tmp_path,
        "Norder=2/Dir=0/Npix=15/Norder=3/Dir=0/Npix=0.parquet"
    )

    assert file_io.does_file_or_directory_exist(path)

    res_df = pd.read_parquet(path)

    assert len(res_df) == 317

    cols = res_df.columns.values.tolist()

    drop_cols = [
        "partition_order", 
        "partition_pixel", 
        "margin_check", 
        "margin_pixel",
        "is_trunc"
    ]

    for col in drop_cols:
        assert col not in cols
