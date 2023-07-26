"""Test resume file operations"""

import pytest

from hipscat_import.pipeline_resume_plan import PipelineResumePlan


def test_log_key(tmp_path):
    """Verify expected behavior of logging key file"""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    log_file = "log_file.log"

    ## key file should start out empty
    assert len(plan.read_log_keys(log_file)) == 0

    ## Write a single key and verify we can retrieve it
    plan.write_log_key(log_file, "abc")
    keys = plan.read_log_keys(log_file)
    assert keys == ["abc"]

    ## add a second key, and verify we retrieve both.
    plan.write_log_key(log_file, "def")
    keys = plan.read_log_keys(log_file)
    assert keys == ["abc", "def"]

    ## clear all the resume files and verify there are no more keys.
    plan.clean_resume_files()

    assert len(plan.read_log_keys(log_file)) == 0


def test_done_file(tmp_path):
    """Verify expected behavior of done file"""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    done_file = "action_done"

    ## done file should not exist
    assert not plan.done_file_exists(done_file)

    ## mark done and check that we can find it
    plan.touch_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## mark done is idempotent
    plan.touch_done_file(done_file)
    assert plan.done_file_exists(done_file)

    ## clear all the resume files and verify done file is gone.
    plan.clean_resume_files()

    assert not plan.done_file_exists(done_file)


def test_safe_to_resume(tmp_path):
    """Check that we throw errors when it's not safe to resume."""
    plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False)
    plan.safe_to_resume()

    ## check is idempotent - intermediate directory exists but does not
    ## contain files.
    plan.safe_to_resume()

    ## take some action and verify that we can no longer resume without
    ## explicitly setting resume=True
    done_file = "action_done"
    plan.touch_done_file(done_file)
    with pytest.raises(ValueError, match="contains intermediate"):
        plan.safe_to_resume()

    ## If we mark as a resuming pipeline, we're safe to resume.
    resuming_plan = PipelineResumePlan(tmp_path=tmp_path, progress_bar=False, resume=True)
    resuming_plan.safe_to_resume()

    ## If there are no more intermediate files, we don't need to set resume.
    plan.clean_resume_files()
    plan.safe_to_resume()
