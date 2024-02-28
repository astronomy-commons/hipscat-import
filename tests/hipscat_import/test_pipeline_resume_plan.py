"""Test resume file operations"""

import os
from pathlib import Path

import pytest

from hipscat_import.pipeline_resume_plan import PipelineResumePlan


def test_done_key(tmp_path):
    """Verify expected behavior of marking stage progress via done files."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    stage = "testing"
    os.makedirs(os.path.join(tmp_path, stage))

    keys = plan.read_done_keys(stage)
    assert len(keys) == 0

    PipelineResumePlan.touch_key_done_file(tmp_path, stage, "key_01")
    keys = plan.read_done_keys(stage)
    assert keys == ["key_01"]

    PipelineResumePlan.touch_key_done_file(tmp_path, stage, "key_02")
    keys = plan.read_done_keys(stage)
    assert keys == ["key_01", "key_02"]

    plan.clean_resume_files()

    keys = plan.read_done_keys(stage)
    assert len(keys) == 0


def test_done_file(tmp_path):
    """Verify expected behavior of done file"""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    done_file = "action_done"

    ## done file should not exist
    assert not plan.done_file_exists(done_file)

    ## mark done and check that we can find it
    plan.touch_stage_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## mark done is idempotent
    plan.touch_stage_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## clear all the resume files and verify done file is gone.
    plan.clean_resume_files()

    assert not plan.done_file_exists(done_file)


def test_get_keys_from_results(tmp_path):
    """Test that we can create a list of completed keys via the output results files."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)
    keys = PipelineResumePlan.get_keys_from_file_names(tmp_path, ".foo")
    assert len(keys) == 0

    Path(tmp_path, "file_0.foo").touch()
    keys = PipelineResumePlan.get_keys_from_file_names(tmp_path, ".foo")
    assert keys == ["file_0"]

    Path(tmp_path, "file_1.foo").touch()
    keys = PipelineResumePlan.get_keys_from_file_names(tmp_path, ".foo")
    assert keys == ["file_0", "file_1"]

    plan.clean_resume_files()
    keys = PipelineResumePlan.get_keys_from_file_names(tmp_path, ".foo")
    assert len(keys) == 0


def test_safe_to_resume(tmp_path):
    """Check that we throw errors when it's not safe to resume."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)
    plan.safe_to_resume()

    ## check is idempotent - intermediate directory exists but does not
    ## contain files.
    plan.safe_to_resume()

    ## take some action and verify that we can no longer resume without
    ## explicitly setting resume=True
    done_file = "action_done"
    plan.touch_stage_done_file(done_file)
    plan.safe_to_resume()

    ## If we mark as a resuming pipeline, we're safe to resume.
    resuming_plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    resuming_plan.safe_to_resume()

    ## If there are no more intermediate files, we don't need to set resume.
    plan.clean_resume_files()
    plan.safe_to_resume()


@pytest.mark.dask
def test_wait_for_futures(tmp_path, dask_client):
    """Test that we can wait around for futures to complete."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=False)

    def error_on_even(argument):
        """Silly little method used to test futures that fail under predictable conditions"""
        if argument % 2 == 0:
            raise RuntimeError("we are at odds with evens")

    ## Everything is fine if we're all odd
    futures = [dask_client.submit(error_on_even, 1)]
    plan.wait_for_futures(futures, "test")

    ## Throw an even in the mix, and we'll see some stages fail. Should cause whole stage to fail.
    futures = [dask_client.submit(error_on_even, 1), dask_client.submit(error_on_even, 2)]
    with pytest.raises(RuntimeError, match="Some test stages failed"):
        plan.wait_for_futures(futures, "test")


def test_formatted_stage_name():
    """Test that we make pretty stage names for presenting in progress bars"""
    formatted = PipelineResumePlan.get_formatted_stage_name(None)
    assert formatted == "Progress  "

    formatted = PipelineResumePlan.get_formatted_stage_name("")
    assert formatted == "Progress  "

    formatted = PipelineResumePlan.get_formatted_stage_name("stage")
    assert formatted == "Stage     "

    formatted = PipelineResumePlan.get_formatted_stage_name("very long stage name")
    assert formatted == "Very long stage name"
