"""Test components of SOAP"""

import os

import pandas as pd
import pytest
from hipscat.catalog import Catalog
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.soap.resume_plan import SoapPlan, source_to_object_map


def test_source_to_object_map(small_sky_object_catalog, small_sky_source_catalog, small_sky_soap_maps):
    """Test creating plan map for object and source catalogs."""
    object_catalog = Catalog.read_from_hipscat(small_sky_object_catalog)
    source_catalog = Catalog.read_from_hipscat(small_sky_source_catalog)

    source_to_object = source_to_object_map(object_catalog, source_catalog)
    assert source_to_object == small_sky_soap_maps


def test_object_to_source_map(small_sky_object_catalog, small_sky_source_catalog):
    """Test creating plan map for object and source catalogs."""
    expected = {
        HealpixPixel(0, 11): [
            HealpixPixel(2, 176),
            HealpixPixel(2, 177),
            HealpixPixel(2, 178),
            HealpixPixel(2, 179),
            HealpixPixel(2, 180),
            HealpixPixel(2, 181),
            HealpixPixel(2, 182),
            HealpixPixel(2, 183),
            HealpixPixel(2, 184),
            HealpixPixel(2, 185),
            HealpixPixel(2, 186),
            HealpixPixel(2, 187),
            HealpixPixel(1, 47),
            HealpixPixel(0, 4),
        ]
    }
    ## Oh, we're so silly!
    object_catalog = Catalog.read_from_hipscat(small_sky_source_catalog)
    source_catalog = Catalog.read_from_hipscat(small_sky_object_catalog)

    source_to_object = source_to_object_map(object_catalog, source_catalog)
    assert source_to_object == expected


def test_mismatch_order_map(catalog_info_data, source_catalog_info):
    """Create some catalogs that will exercise edge case behavior of map-generation."""
    object_catalog = Catalog(
        CatalogInfo(**catalog_info_data),
        [
            HealpixPixel(1, 16),
            HealpixPixel(2, 68),
            HealpixPixel(2, 69),
            HealpixPixel(2, 70),
            HealpixPixel(2, 71),
        ],
    )
    source_catalog = Catalog(CatalogInfo(**source_catalog_info), [HealpixPixel(1, 16)])

    expected = {
        HealpixPixel(1, 16): [
            HealpixPixel(1, 16),
            HealpixPixel(2, 71),
            HealpixPixel(2, 68),
            HealpixPixel(2, 69),
            HealpixPixel(2, 70),
        ],
    }
    source_to_object = source_to_object_map(object_catalog, source_catalog)
    assert source_to_object == expected


def test_counting_done(small_sky_soap_args):
    """Verify expected behavior of counting done file"""
    plan = SoapPlan(small_sky_soap_args)
    assert not plan.is_counting_done()
    plan.touch_done_file(SoapPlan.COUNTING_STAGE)
    assert plan.is_counting_done()

    plan.clean_resume_files()
    assert not plan.is_counting_done()


def test_count_keys(small_sky_soap_args):
    """Verify expected behavior of counting keys file"""
    plan = SoapPlan(small_sky_soap_args)
    assert len(plan.count_keys) == 14

    ## Mark one done and check that there's one less key to count later.
    plan.write_log_key(SoapPlan.COUNTING_STAGE, "2_187")

    plan.gather_plan(small_sky_soap_args)
    assert len(plan.count_keys) == 13

    ## Mark them ALL done and check that there are on keys later.
    plan.touch_done_file(SoapPlan.COUNTING_STAGE)

    plan.gather_plan(small_sky_soap_args)
    assert len(plan.count_keys) == 0


@pytest.mark.timeout(2)
def test_cached_map_file(small_sky_soap_args):
    """Verify that we cache the mapping file for later use.
    This can be expensive to compute for large survey cross-products!"""
    plan = SoapPlan(small_sky_soap_args)
    assert len(plan.count_keys) == 14

    ## The source partition mapping should be cached in a file.
    cache_map_file = os.path.join(small_sky_soap_args.tmp_path, SoapPlan.SOURCE_MAP_FILE)
    assert os.path.exists(cache_map_file)

    plan = SoapPlan(small_sky_soap_args)
    assert len(plan.count_keys) == 14
