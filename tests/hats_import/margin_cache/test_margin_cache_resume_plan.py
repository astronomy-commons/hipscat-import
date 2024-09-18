import numpy as np
import numpy.testing as npt
import pytest
from hats.catalog import Catalog

from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.margin_cache.margin_cache_resume_plan import (
    MarginCachePlan,
    _find_partition_margin_pixel_pairs,
)

# pylint: disable=protected-access


@pytest.fixture
def small_sky_margin_args(tmp_path, small_sky_object_catalog):
    return MarginCacheArguments(
        margin_threshold=5.0,
        margin_order=4,
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="catalog_cache",
        progress_bar=False,
        resume=True,
    )


def test_done_checks(small_sky_margin_args):
    """Verify that done files imply correct pipeline execution order:
    mapping > reducing
    """
    plan = MarginCachePlan(small_sky_margin_args)
    plan.touch_stage_done_file(MarginCachePlan.REDUCING_STAGE)

    with pytest.raises(ValueError, match="before reducing"):
        plan._gather_plan(small_sky_margin_args)

    plan.touch_stage_done_file(MarginCachePlan.MAPPING_STAGE)
    plan._gather_plan(small_sky_margin_args)
    assert plan.is_mapping_done()
    assert plan.is_reducing_done()

    plan.clean_resume_files()

    plan = MarginCachePlan(small_sky_margin_args)
    plan.touch_stage_done_file(MarginCachePlan.MAPPING_STAGE)
    plan._gather_plan(small_sky_margin_args)

    assert plan.is_mapping_done()
    assert not plan.is_reducing_done()


def never_fails():
    """Method never fails, but never marks intermediate success file."""
    return


@pytest.mark.dask
def test_some_map_task_failures(small_sky_margin_args, dask_client):
    """Test that we only consider map stage successful if all done files are written"""
    plan = MarginCachePlan(small_sky_margin_args)

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="1 mapping stages"):
        plan.wait_for_mapping(futures)

    MarginCachePlan.touch_key_done_file(plan.tmp_path, MarginCachePlan.MAPPING_STAGE, "0_11")

    ## Method succeeds, *and* done files are present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_mapping(futures)


@pytest.mark.dask
def test_some_reducing_task_failures(small_sky_margin_args, dask_client):
    """Test that we only consider reduce stage successful if all done files are written"""
    plan = MarginCachePlan(small_sky_margin_args)

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="12 reducing stages"):
        plan.wait_for_reducing(futures)

    MarginCachePlan.touch_key_done_file(plan.tmp_path, MarginCachePlan.REDUCING_STAGE, "0_11")

    ## Method succeeds, but only *ONE* done file is present.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="11 reducing stages"):
        plan.wait_for_reducing(futures)

    for partition in range(0, 12):
        MarginCachePlan.touch_key_done_file(plan.tmp_path, MarginCachePlan.REDUCING_STAGE, f"0_{partition}")

    ## Method succeeds, *and* done files are present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_reducing(futures)


def test_partition_margin_pixel_pairs(small_sky_source_catalog):
    """Ensure partition_margin_pixel_pairs can generate main partition pixels."""
    source_catalog = Catalog.read_hats(small_sky_source_catalog)
    margin_pairs = _find_partition_margin_pixel_pairs(source_catalog.get_healpix_pixels(), 3)

    expected = np.array([0, 2, 8, 10, 32, 34, 40, 42, 192, 192])

    npt.assert_array_equal(margin_pairs.iloc[:10]["margin_pixel"], expected)
    assert len(margin_pairs) == 196


def test_partition_margin_pixel_pairs_negative(small_sky_source_catalog):
    """Ensure partition_margin_pixel_pairs can generate negative tree pixels."""
    source_catalog = Catalog.read_hats(small_sky_source_catalog)

    partition_stats = source_catalog.get_healpix_pixels()
    negative_pixels = source_catalog.generate_negative_tree_pixels()
    combined_pixels = partition_stats + negative_pixels

    margin_pairs = _find_partition_margin_pixel_pairs(combined_pixels, 3)

    expected_order = 0
    expected_pixel = 10
    expected = np.array([760, 760, 762, 762, 763, 765, 766, 767, 767, 767])

    assert margin_pairs.iloc[218]["partition_order"] == expected_order
    assert margin_pairs.iloc[218]["partition_pixel"] == expected_pixel
    npt.assert_array_equal(margin_pairs.iloc[-10:]["margin_pixel"], expected)
    assert len(margin_pairs) == 536
