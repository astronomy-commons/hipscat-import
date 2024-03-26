"""Tests of map reduce operations"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.catalog import PartitionInfo
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset
from hipscat.io import file_io, paths

import hipscat_import.margin_cache.margin_cache as mc
from hipscat_import.margin_cache.margin_cache_arguments import MarginCacheArguments

# pylint: disable=protected-access


@pytest.mark.dask(timeout=150)
def test_margin_cache_gen(small_sky_source_catalog, tmp_path, dask_client):
    """Test that margin cache generation works end to end."""
    args = MarginCacheArguments(
        margin_threshold=180.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=8,
        progress_bar=False,
    )

    assert args.catalog.catalog_info.ra_column == "source_ra"

    mc.generate_margin_cache(args, dask_client)

    norder = 1
    npix = 47

    test_file = paths.pixel_catalog_file(args.catalog_path, norder, npix)

    data = pd.read_parquet(test_file)

    assert len(data) == 13

    assert all(data[PartitionInfo.METADATA_ORDER_COLUMN_NAME] == norder)
    assert all(data[PartitionInfo.METADATA_PIXEL_COLUMN_NAME] == npix)
    assert all(data[PartitionInfo.METADATA_DIR_COLUMN_NAME] == int(npix / 10000) * 10000)

    assert data.dtypes[PartitionInfo.METADATA_ORDER_COLUMN_NAME] == np.uint8
    assert data.dtypes[PartitionInfo.METADATA_DIR_COLUMN_NAME] == np.uint64
    assert data.dtypes[PartitionInfo.METADATA_PIXEL_COLUMN_NAME] == np.uint64

    npt.assert_array_equal(
        data.columns,
        [
            "source_id",
            "source_ra",
            "source_dec",
            "mjd",
            "mag",
            "band",
            "object_id",
            "object_ra",
            "object_dec",
            "Norder",
            "Dir",
            "Npix",
            "margin_Norder",
            "margin_Dir",
            "margin_Npix",
        ],
    )
    assert data.index.name == "_hipscat_index"

    catalog = HealpixDataset.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path


@pytest.mark.dask(timeout=150)
def test_margin_cache_gen_negative_pixels(small_sky_source_catalog, tmp_path, dask_client):
    """Test that margin cache generation can generate a file for a negative pixel."""
    with pytest.warns(UserWarning, match="smaller resolution"):
        args = MarginCacheArguments(
            margin_threshold=36000.0,
            input_catalog_path=small_sky_source_catalog,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
            margin_order=4,
            progress_bar=False,
        )

    assert args.catalog.catalog_info.ra_column == "source_ra"

    mc.generate_margin_cache(args, dask_client)

    norder = 0
    npix = 7

    negative_test_file = paths.pixel_catalog_file(args.catalog_path, norder, npix)

    negative_data = pd.read_parquet(negative_test_file)

    assert len(negative_data) > 0


def test_partition_margin_pixel_pairs(small_sky_source_catalog, tmp_path):
    """Ensure partition_margin_pixel_pairs can generate main partition pixels."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
    )

    margin_pairs = mc._find_partition_margin_pixel_pairs(
        args.catalog.partition_info.get_healpix_pixels(), args.margin_order
    )

    expected = np.array([725, 733, 757, 765, 727, 735, 759, 767, 469, 192])

    npt.assert_array_equal(margin_pairs.iloc[:10]["margin_pixel"], expected)
    assert len(margin_pairs) == 196


def test_partition_margin_pixel_pairs_negative(small_sky_source_catalog, tmp_path):
    """Ensure partition_margin_pixel_pairs can generate negative tree pixels."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
    )

    partition_stats = args.catalog.partition_info.get_healpix_pixels()
    negative_pixels = args.catalog.generate_negative_tree_pixels()
    combined_pixels = partition_stats + negative_pixels

    margin_pairs = mc._find_partition_margin_pixel_pairs(combined_pixels, args.margin_order)

    expected_order = 0
    expected_pixel = 10
    expected = np.array([490, 704, 712, 736, 744, 706, 714, 738, 746, 512])

    assert margin_pairs.iloc[-1]["partition_order"] == expected_order
    assert margin_pairs.iloc[-1]["partition_pixel"] == expected_pixel
    npt.assert_array_equal(margin_pairs.iloc[-10:]["margin_pixel"], expected)
    assert len(margin_pairs) == 536


def test_create_margin_directory(small_sky_source_catalog, tmp_path):
    """Ensure create_margin_directory works on main partition_pixels"""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
    )

    mc._create_margin_directory(
        stats=args.catalog.partition_info.get_healpix_pixels(),
        output_path=args.catalog_path,
        storage_options=None,
    )

    output = file_io.append_paths_to_pointer(args.catalog_path, "Norder=0", "Dir=0")
    assert file_io.does_file_or_directory_exist(output)
