"""Tests of map reduce operations"""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.io import file_io, paths

import hipscat_import.margin_cache.margin_cache as mc
from hipscat_import.margin_cache import MarginCacheArguments

# pylint: disable=protected-access


@pytest.mark.dask(timeout=20)
def test_margin_cache_gen(small_sky_source_catalog, tmp_path, dask_client):
    """Test that margin cache generation works end to end."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
    )

    assert args.catalog.catalog_info.ra_column == "source_ra"

    mc.generate_margin_cache(args, dask_client)

    print(args.catalog.partition_info.get_healpix_pixels())

    norder = 1
    npix = 47

    test_file = paths.pixel_catalog_file(
        args.catalog_path, norder, npix
    )

    data = pd.read_parquet(test_file)

    assert len(data) == 4

def test_partition_margin_pixel_pairs(small_sky_source_catalog, tmp_path):
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
    )

    margin_pairs = mc._find_partition_margin_pixel_pairs(
        args.catalog.partition_info.get_healpix_pixels(), args.margin_order
    )

    expected = np.array([725, 733, 757, 765, 727, 735, 759, 767, 469, 192])

    npt.assert_array_equal(margin_pairs.iloc[:10]["margin_pixel"], expected)
    assert len(margin_pairs) == 196


def test_create_margin_directory(small_sky_source_catalog, tmp_path):
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
    )

    mc._create_margin_directory(
        stats=args.catalog.partition_info.get_healpix_pixels(),
        output_path=args.catalog_path,
    )

    output = file_io.append_paths_to_pointer(args.catalog_path, "Norder=0", "Dir=0")
    assert file_io.does_file_or_directory_exist(output)
