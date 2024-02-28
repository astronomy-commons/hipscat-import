"""Test components of SOAP"""

import os
from pathlib import Path

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
    plan.touch_stage_done_file(SoapPlan.COUNTING_STAGE)
    assert plan.is_counting_done()

    plan.clean_resume_files()
    assert not plan.is_counting_done()


def test_count_keys(small_sky_soap_args):
    """Verify expected behavior of counting keys file"""
    plan = SoapPlan(small_sky_soap_args)
    assert len(plan.count_keys) == 14

    ## Mark one done and check that there's one less key to count later.
    Path(small_sky_soap_args.tmp_path, "2_187.csv").touch()

    with pytest.warns(UserWarning, match="resuming prior progress"):
        plan.gather_plan(small_sky_soap_args)
    assert len(plan.count_keys) == 13

    ## Mark them ALL done and check that there are on keys later.
    plan.touch_stage_done_file(SoapPlan.COUNTING_STAGE)

    with pytest.warns(UserWarning, match="resuming prior progress"):
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

    with pytest.warns(UserWarning, match="resuming prior progress"):
        plan = SoapPlan(small_sky_soap_args)
    assert len(plan.count_keys) == 14


def test_get_sources_to_count(small_sky_soap_args):
    """Test generation of remaining count items"""
    source_pixel_map = {HealpixPixel(0, 11): (131, [44, 45, 46])}
    plan = SoapPlan(small_sky_soap_args)

    ## Kind of silly, but clear out the pixel map, since it's populated on init.
    ## Fail to find the remaining sources to count because we don't know the map.
    plan.source_pixel_map = None
    with pytest.raises(ValueError, match="source_pixel_map"):
        remaining_count_items = plan.get_sources_to_count()

    ## Can now successfully find sources to count.
    remaining_count_items = plan.get_sources_to_count(source_pixel_map=source_pixel_map)
    assert len(remaining_count_items) == 1

    ## Use previous value of sources map, and find intermediate file, so there are no
    ## remaining sources to count.
    Path(small_sky_soap_args.tmp_path, "0_11.csv").touch()
    remaining_count_items = plan.get_sources_to_count()
    assert len(remaining_count_items) == 0


def never_fails():
    """Method never fails, but never marks intermediate success file."""
    return


@pytest.mark.dask
def test_some_counting_task_failures(small_sky_soap_args, dask_client):
    """Test that we only consider counting stage successful if all done files are written"""
    plan = SoapPlan(small_sky_soap_args)

    ## Method doesn't FAIL, but it doesn't write out the intermediate results file either.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="14 counting stages"):
        plan.wait_for_counting(futures)

    ## Write one intermediate results file. There are fewer unsuccessful stages.
    Path(small_sky_soap_args.tmp_path, "2_187.csv").touch()
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="13 counting stages"):
        plan.wait_for_counting(futures)

    ## Write ALL the intermediate results files. Waiting for results will succeed.
    for _, _, count_key in plan.count_keys:
        Path(small_sky_soap_args.tmp_path, f"{count_key}.csv").touch()
    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_counting(futures)


@pytest.mark.dask
def test_some_reducing_task_failures(small_sky_soap_args, dask_client):
    """Test that we only consider reducing stage successful if all done files are written"""
    plan = SoapPlan(small_sky_soap_args)

    ## Method doesn't FAIL, but it doesn't write out the intermediate results file either.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="1 reducing stages"):
        plan.wait_for_reducing(futures)

    ## Write ALL the intermediate results files. Waiting for results will succeed.
    Path(small_sky_soap_args.tmp_path, "reducing", "0_11_done").touch()

    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_reducing(futures)
