"""Test components of SOAP"""

import os

import pytest

from hipscat_import.soap.resume_plan import SoapPlan, source_to_object_map


def test_source_to_object_map(small_sky_soap_args, small_sky_soap_maps):
    """Test creating plan map for object and source catalogs."""
    source_to_object = source_to_object_map(small_sky_soap_args)
    assert source_to_object == small_sky_soap_maps


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
