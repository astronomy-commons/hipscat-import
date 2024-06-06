"""Tests of margin cache generation arguments"""

import pytest
from hipscat.io import write_metadata
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.margin_cache.margin_cache_arguments import MarginCacheArguments


def test_empty_required(tmp_path):
    """*Most* required arguments are provided."""
    ## Input catalog path is missing
    with pytest.raises(ValueError, match="input_catalog_path"):
        MarginCacheArguments(
            margin_threshold=5.0,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
        )

    ## Input catalog path is bad
    with pytest.raises(ValueError, match="input_catalog_path"):
        MarginCacheArguments(
            input_catalog_path="/foo",
            margin_threshold=5.0,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
        )


def test_margin_order_dynamic(small_sky_source_catalog, tmp_path):
    """Ensure we can dynamically set the margin_order"""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
    )

    assert args.margin_order == 3


def test_margin_order_static(small_sky_source_catalog, tmp_path):
    """Ensure we can manually set the margin_order"""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=4,
    )

    assert args.margin_order == 4


def test_margin_order_invalid(small_sky_source_catalog, tmp_path):
    """Ensure we raise an exception when margin_order is invalid"""
    with pytest.raises(ValueError, match="margin_order"):
        MarginCacheArguments(
            margin_threshold=5.0,
            input_catalog_path=small_sky_source_catalog,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
            margin_order=2,
        )


def test_debug_filter_pixel_list(small_sky_source_catalog, tmp_path):
    """Ensure we can generate catalog with a filtereed list of pixels, and
    that we raise an exception when the filter results in an empty catalog."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=4,
        debug_filter_pixel_list=[HealpixPixel(0, 11)],
    )

    assert len(args.catalog.get_healpix_pixels()) == 13

    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=4,
        debug_filter_pixel_list=[HealpixPixel(1, 44)],
    )

    assert len(args.catalog.get_healpix_pixels()) == 4

    with pytest.raises(ValueError, match="debug_filter_pixel_list"):
        MarginCacheArguments(
            margin_threshold=5.0,
            input_catalog_path=small_sky_source_catalog,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
            margin_order=4,
            debug_filter_pixel_list=[HealpixPixel(0, 5)],
        )


def test_margin_threshold_warns(small_sky_source_catalog, tmp_path):
    """Ensure we give a warning when margin_threshold is greater than margin_order resolution"""

    with pytest.warns(UserWarning, match="margin_threshold"):
        MarginCacheArguments(
            margin_threshold=360.0,
            input_catalog_path=small_sky_source_catalog,
            output_path=tmp_path,
            output_artifact_name="catalog_cache",
            margin_order=16,
        )


def test_to_catalog_info(small_sky_source_catalog, tmp_path):
    """Verify creation of catalog info for margin cache to be created."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=4,
    )
    catalog_info = args.to_catalog_info(total_rows=10)
    assert catalog_info.catalog_name == args.output_artifact_name
    assert catalog_info.total_rows == 10


def test_provenance_info(small_sky_source_catalog, tmp_path):
    """Verify that provenance info includes margin-cache-specific fields."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        margin_order=4,
        debug_filter_pixel_list=[HealpixPixel(1, 44)],
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "margin_threshold" in runtime_args

    write_metadata.write_provenance_info(
        catalog_base_dir=args.catalog_path, dataset_info=args.to_catalog_info(20_000), tool_args=runtime_args
    )
