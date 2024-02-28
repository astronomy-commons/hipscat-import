"""Test catalog resume logic"""

import hipscat.pixel_math as hist
import numpy.testing as npt
import pytest
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.catalog.resume_plan import ResumePlan


def test_mapping_done(tmp_path):
    """Verify expected behavior of mapping done file"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)
    assert not plan.is_mapping_done()
    plan.touch_stage_done_file(ResumePlan.MAPPING_STAGE)
    assert plan.is_mapping_done()

    plan.clean_resume_files()
    assert not plan.is_mapping_done()


def test_reducing_done(tmp_path):
    """Verify expected behavior of reducing done file"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)
    assert not plan.is_reducing_done()
    plan.touch_stage_done_file(ResumePlan.REDUCING_STAGE)
    assert plan.is_reducing_done()

    plan.clean_resume_files()
    assert not plan.is_reducing_done()


def test_done_checks(tmp_path):
    """Verify that done files imply correct pipeline execution order:
    mapping > splitting > reducing
    """
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_stage_done_file(ResumePlan.REDUCING_STAGE)

    with pytest.warns(UserWarning, match="resuming prior progress"):
        with pytest.raises(ValueError, match="before reducing"):
            plan.gather_plan()

    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    with pytest.warns(UserWarning, match="resuming prior progress"):
        with pytest.raises(ValueError, match="before reducing"):
            plan.gather_plan()

    plan.clean_resume_files()

    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    with pytest.warns(UserWarning, match="resuming prior progress"):
        with pytest.raises(ValueError, match="before splitting"):
            plan.gather_plan()


def test_same_input_paths(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can only resume if the input_paths are the same."""
    plan = ResumePlan(
        tmp_path=tmp_path,
        progress_bar=False,
        resume=True,
        input_paths=[small_sky_single_file, formats_headers_csv],
    )
    map_files = plan.map_files
    assert len(map_files) == 2

    with pytest.warns(UserWarning, match="resuming prior progress"):
        with pytest.raises(ValueError, match="Different file set"):
            ResumePlan(
                tmp_path=tmp_path,
                progress_bar=False,
                resume=True,
                input_paths=[small_sky_single_file],
            )

    ## List is the same length, but includes a duplicate
    with pytest.warns(UserWarning, match="resuming prior progress"):
        with pytest.raises(ValueError, match="Different file set"):
            ResumePlan(
                tmp_path=tmp_path,
                progress_bar=False,
                resume=True,
                input_paths=[small_sky_single_file, small_sky_single_file],
            )

    ## Includes a duplicate file, but that's ok.
    with pytest.warns(UserWarning, match="resuming prior progress"):
        plan = ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file, small_sky_single_file, formats_headers_csv],
        )
    map_files = plan.map_files
    assert len(map_files) == 2


def test_read_write_histogram(tmp_path):
    """Test that we can read what we write into a histogram file."""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## We're not ready to read the final histogram - missing partial histograms.
    with pytest.raises(RuntimeError, match="map stages"):
        result = plan.read_histogram(0)

    expected = hist.empty_histogram(0)
    expected[11] = 131

    remaining_keys = plan.get_remaining_map_keys()
    assert remaining_keys == [("map_0", "foo1")]

    ResumePlan.write_partial_histogram(tmp_path=tmp_path, mapping_key="map_0", histogram=expected)

    remaining_keys = plan.get_remaining_map_keys()
    assert len(remaining_keys) == 0
    result = plan.read_histogram(0)
    npt.assert_array_equal(result, expected)


def never_fails():
    """Method never fails, but never marks intermediate success file."""
    return


@pytest.mark.dask
def test_some_map_task_failures(tmp_path, dask_client):
    """Test that we only consider map stage successful if all partial files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## Method doesn't FAIL, but it doesn't write out the partial histogram either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="map stages"):
        plan.wait_for_mapping(futures)

    expected = hist.empty_histogram(0)
    expected[11] = 131

    ResumePlan.write_partial_histogram(tmp_path=tmp_path, mapping_key="map_0", histogram=expected)

    ## Method succeeds, *and* partial histogram is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_mapping(futures)


def test_read_write_splitting_keys(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can read what we write into a reducing log file."""
    input_paths = [small_sky_single_file, formats_headers_csv]
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True, input_paths=input_paths)
    split_keys = plan.split_keys
    assert len(split_keys) == 2

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_0")

    with pytest.warns(UserWarning, match="resuming prior progress"):
        plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 1
    assert split_keys[0][0] == "split_1"

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_1")
    with pytest.warns(UserWarning, match="resuming prior progress"):
        plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 0

    plan.clean_resume_files()
    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 2


@pytest.mark.dask
def test_some_split_task_failures(tmp_path, dask_client):
    """Test that we only consider split stage successful if all done files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, input_paths=["foo1"])

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="split stages"):
        plan.wait_for_splitting(futures)

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.SPLITTING_STAGE, "split_0")

    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_splitting(futures)


def test_get_reduce_items(tmp_path):
    """Test generation of remaining reduce items"""
    destination_pixel_map = {HealpixPixel(0, 11): (131, [44, 45, 46])}
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)

    with pytest.raises(RuntimeError, match="destination pixel map"):
        remaining_reduce_items = plan.get_reduce_items()

    remaining_reduce_items = plan.get_reduce_items(destination_pixel_map=destination_pixel_map)
    assert len(remaining_reduce_items) == 1

    ResumePlan.reducing_key_done(tmp_path=tmp_path, reducing_key="0_11")
    remaining_reduce_items = plan.get_reduce_items(destination_pixel_map=destination_pixel_map)
    assert len(remaining_reduce_items) == 0


@pytest.mark.dask
def test_some_reduce_task_failures(tmp_path, dask_client):
    """Test that we only consider reduce stage successful if all done files are written"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)

    destination_pixel_map = {HealpixPixel(0, 11): (131, [44, 45, 46])}
    remaining_reduce_items = plan.get_reduce_items(destination_pixel_map=destination_pixel_map)
    assert len(remaining_reduce_items) == 1

    ## Method doesn't FAIL, but it doesn't write out the done file either.
    ## Since the intermediate files aren't found, we throw an error.
    futures = [dask_client.submit(never_fails)]
    with pytest.raises(RuntimeError, match="reduce stages"):
        plan.wait_for_reducing(futures)

    ResumePlan.touch_key_done_file(tmp_path, ResumePlan.REDUCING_STAGE, "0_11")

    ## Method succeeds, and done file is present.
    futures = [dask_client.submit(never_fails)]
    plan.wait_for_reducing(futures)
