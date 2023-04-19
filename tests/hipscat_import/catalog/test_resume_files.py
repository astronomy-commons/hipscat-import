"""Test resume file operations"""

import hipscat.pixel_math as hist
import numpy.testing as npt
import pytest

from hipscat_import.catalog.resume_files import (
    clean_resume_files,
    is_mapping_done,
    is_reducing_done,
    read_histogram,
    read_mapping_keys,
    read_reducing_keys,
    set_mapping_done,
    set_reducing_done,
    write_histogram,
    write_mapping_done_key,
    write_mapping_start_key,
    write_reducing_key,
)


def test_mapping_done(tmp_path):
    """Verify expected behavior of mapping done file"""
    assert not is_mapping_done(tmp_path)
    set_mapping_done(tmp_path)
    assert is_mapping_done(tmp_path)

    clean_resume_files(tmp_path)
    assert not is_mapping_done(tmp_path)


def test_reducing_done(tmp_path):
    """Verify expected behavior of reducing done file"""
    assert not is_reducing_done(tmp_path)
    set_reducing_done(tmp_path)
    assert is_reducing_done(tmp_path)

    clean_resume_files(tmp_path)
    assert not is_reducing_done(tmp_path)


def test_read_write_histogram(tmp_path):
    """Test that we can read what we write into a histogram file."""
    empty = hist.empty_histogram(0)
    result = read_histogram(tmp_path, 0)
    npt.assert_array_equal(result, empty)

    expected = hist.empty_histogram(0)
    expected[11] = 131
    write_histogram(tmp_path, expected)
    result = read_histogram(tmp_path, 0)
    npt.assert_array_equal(result, expected)

    clean_resume_files(tmp_path)
    result = read_histogram(tmp_path, 0)
    npt.assert_array_equal(result, empty)


def test_read_write_mapping_keys(tmp_path):
    """Test that we can read what we write into a mapping log file."""
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 0

    write_mapping_start_key(tmp_path, "key1")
    write_mapping_done_key(tmp_path, "key1")
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 1
    assert mapping_keys[0] == "key1"

    write_mapping_start_key(tmp_path, "key2")
    write_mapping_done_key(tmp_path, "key2")
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 2
    assert mapping_keys[0] == "key1"
    assert mapping_keys[1] == "key2"

    clean_resume_files(tmp_path)
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 0


def test_read_write_mapping_keys_corrupt(tmp_path):
    """Test that we can read what we write into a mapping log file."""
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 0

    write_mapping_start_key(tmp_path, "key1")
    write_mapping_done_key(tmp_path, "key1")
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 1
    assert mapping_keys[0] == "key1"

    write_mapping_start_key(tmp_path, "key2")
    with pytest.raises(ValueError, match="logs are corrupted"):
        read_mapping_keys(tmp_path)

    clean_resume_files(tmp_path)
    mapping_keys = read_mapping_keys(tmp_path)
    assert len(mapping_keys) == 0


def test_read_write_reducing_keys(tmp_path):
    """Test that we can read what we write into a reducing log file."""
    reducing_keys = read_reducing_keys(tmp_path)
    assert len(reducing_keys) == 0

    write_reducing_key(tmp_path, "key_1")
    reducing_keys = read_reducing_keys(tmp_path)
    assert len(reducing_keys) == 1
    assert reducing_keys[0] == "key_1"

    write_reducing_key(tmp_path, "key_2")
    reducing_keys = read_reducing_keys(tmp_path)
    assert len(reducing_keys) == 2
    assert reducing_keys[0] == "key_1"
    assert reducing_keys[1] == "key_2"

    clean_resume_files(tmp_path)
    reducing_keys = read_reducing_keys(tmp_path)
    assert len(reducing_keys) == 0
