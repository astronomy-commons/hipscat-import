"""Test catalog resume logic"""

import hipscat.pixel_math as hist
import numpy.testing as npt
import pytest

from hipscat_import.catalog.resume_plan import ResumePlan


def test_mapping_done(tmp_path):
    """Verify expected behavior of mapping done file"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)
    assert not plan.is_mapping_done()
    plan.touch_done_file(ResumePlan.MAPPING_STAGE)
    assert plan.is_mapping_done()

    plan.clean_resume_files()
    assert not plan.is_mapping_done()


def test_reducing_done(tmp_path):
    """Verify expected behavior of reducing done file"""
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)
    assert not plan.is_reducing_done()
    plan.touch_done_file(ResumePlan.REDUCING_STAGE)
    assert plan.is_reducing_done()

    plan.clean_resume_files()
    assert not plan.is_reducing_done()


def test_done_checks(tmp_path):
    """Verify that done files imply correct pipeline execution order:
    mapping > splitting > reducing
    """
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_done_file(ResumePlan.REDUCING_STAGE)

    with pytest.raises(ValueError, match="before reducing"):
        plan.gather_plan()

    plan.touch_done_file(ResumePlan.SPLITTING_STAGE)
    with pytest.raises(ValueError, match="before reducing"):
        plan.gather_plan()

    plan.clean_resume_files()

    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    plan.touch_done_file(ResumePlan.SPLITTING_STAGE)
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

    with pytest.raises(ValueError, match="Different file set"):
        ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file],
        )

    ## List is the same length, but includes a duplicate
    with pytest.raises(ValueError, match="Different file set"):
        ResumePlan(
            tmp_path=tmp_path,
            progress_bar=False,
            resume=True,
            input_paths=[small_sky_single_file, small_sky_single_file],
        )

    ## Includes a duplicate file, but that's ok.
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
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)

    expected = hist.empty_histogram(0)
    expected[11] = 131

    ResumePlan.write_partial_histogram(tmp_path=tmp_path, mapping_key="map_0", histogram=expected)
    result = plan.read_histogram(0)
    npt.assert_array_equal(result, expected)

    plan.clean_resume_files()

    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False)
    empty = hist.empty_histogram(0)
    result = plan.read_histogram(0)
    npt.assert_array_equal(result, empty)


def test_read_write_map_files(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can list the remaining files to map."""
    input_paths = [small_sky_single_file, formats_headers_csv]
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True, input_paths=input_paths)
    map_files = plan.map_files
    assert len(map_files) == 2

    plan.write_log_key(ResumePlan.MAPPING_STAGE, "map_0")

    plan.gather_plan()
    map_files = plan.map_files
    assert len(map_files) == 1
    assert map_files[0][1] == formats_headers_csv

    plan.write_log_key(ResumePlan.MAPPING_STAGE, "map_1")

    ## Nothing left to map
    plan.gather_plan()
    map_files = plan.map_files
    assert len(map_files) == 0

    plan.clean_resume_files()

    ## No progress to resume - all left to map.
    plan.gather_plan()
    map_files = plan.map_files
    assert len(map_files) == 2


def test_read_write_splitting_keys(tmp_path, small_sky_single_file, formats_headers_csv):
    """Test that we can read what we write into a reducing log file."""
    input_paths = [small_sky_single_file, formats_headers_csv]
    plan = ResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True, input_paths=input_paths)
    split_keys = plan.split_keys
    assert len(split_keys) == 2

    plan.write_log_key(ResumePlan.SPLITTING_STAGE, "split_0")

    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 1
    assert split_keys[0][0] == "split_1"

    plan.write_log_key(ResumePlan.SPLITTING_STAGE, "split_1")
    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 0

    plan.clean_resume_files()
    plan.gather_plan()
    split_keys = plan.split_keys
    assert len(split_keys) == 2
