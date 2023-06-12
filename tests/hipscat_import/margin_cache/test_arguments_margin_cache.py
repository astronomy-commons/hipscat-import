"""Tests of margin cache generation arguments"""


import pytest

from hipscat_import.margin_cache import MarginCacheArguments

# pylint: disable=protected-access


def test_empty_required(tmp_path):
    """*Most* required arguments are provided."""
    ## Input catalog path is missing
    with pytest.raises(FileNotFoundError, match="input_catalog_path"):
        MarginCacheArguments(
            margin_threshold=5.0,
            output_path=tmp_path,
            output_catalog_name="catalog_cache",
        )


def test_margin_order_dynamic(small_sky_source_catalog, tmp_path):
    """Ensure we can dynamically set the margin_order"""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
    )

    assert args.margin_order == 3


def test_margin_order_static(small_sky_source_catalog, tmp_path):
    """Ensure we can manually set the margin_order"""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
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
            output_catalog_name="catalog_cache",
            margin_order=2,
        )


def test_margin_threshold_warns(small_sky_source_catalog, tmp_path):
    """Ensure we give a warning when margin_threshold is greater than margin_order resolution"""

    with pytest.warns(UserWarning, match="margin_threshold"):
        MarginCacheArguments(
            margin_threshold=360.0,
            input_catalog_path=small_sky_source_catalog,
            output_path=tmp_path,
            output_catalog_name="catalog_cache",
            margin_order=16,
        )


def test_to_catalog_info(small_sky_source_catalog, tmp_path):
    """Verify creation of catalog info for margin cache to be created."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
        margin_order=4,
    )
    catalog_info = args.to_catalog_info(total_rows=10)
    assert catalog_info.catalog_name == args.output_catalog_name
    assert catalog_info.total_rows == 10


def test_provenance_info(small_sky_source_catalog, tmp_path):
    """Verify that provenance info includes margin-cache-specific fields."""
    args = MarginCacheArguments(
        margin_threshold=5.0,
        input_catalog_path=small_sky_source_catalog,
        output_path=tmp_path,
        output_catalog_name="catalog_cache",
        margin_order=4,
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "margin_threshold" in runtime_args
