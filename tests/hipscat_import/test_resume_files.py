"""Test resume file operations"""

import tempfile

import hipscat.pixel_math as hist
import numpy.testing as npt

from hipscat_import.resume_files import (
    clean_resume_files,
    is_mapping_done,
    is_reducing_done,
    read_histogram,
    read_mapping_keys,
    read_reducing_keys,
    set_mapping_done,
    set_reducing_done,
    write_histogram,
    write_mapping_key,
    write_reducing_key,
)


def test_mapping_done():
    """Verify expected behavior of mapping done file"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert not is_mapping_done(tmp_dir)
        set_mapping_done(tmp_dir)
        assert is_mapping_done(tmp_dir)

        clean_resume_files(tmp_dir)
        assert not is_mapping_done(tmp_dir)

    assert not is_mapping_done(tmp_dir)


def test_reducing_done():
    """Verify expected behavior of reducing done file"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert not is_reducing_done(tmp_dir)
        set_reducing_done(tmp_dir)
        assert is_reducing_done(tmp_dir)

        clean_resume_files(tmp_dir)
        assert not is_reducing_done(tmp_dir)

    assert not is_reducing_done(tmp_dir)


def test_read_write_histogram():
    """Test that we can read what we write into a histogram file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty = hist.empty_histogram(0)
        result = read_histogram(tmp_dir, 0)
        npt.assert_array_equal(result, empty)

        expected = hist.empty_histogram(0)
        expected[11] = 131
        write_histogram(tmp_dir, expected)
        result = read_histogram(tmp_dir, 0)
        npt.assert_array_equal(result, expected)

        clean_resume_files(tmp_dir)
        result = read_histogram(tmp_dir, 0)
        npt.assert_array_equal(result, empty)


def test_read_write_mapping_keys():
    """Test that we can read what we write into a mapping log file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mapping_keys = read_mapping_keys(tmp_dir)
        assert len(mapping_keys) == 0

        write_mapping_key(tmp_dir, "key1")
        mapping_keys = read_mapping_keys(tmp_dir)
        assert len(mapping_keys) == 1
        assert mapping_keys[0] == "key1"

        write_mapping_key(tmp_dir, "key2")
        mapping_keys = read_mapping_keys(tmp_dir)
        assert len(mapping_keys) == 2
        assert mapping_keys[0] == "key1"
        assert mapping_keys[1] == "key2"

        clean_resume_files(tmp_dir)
        mapping_keys = read_mapping_keys(tmp_dir)
        assert len(mapping_keys) == 0


def test_read_write_reducing_keys():
    """Test that we can read what we write into a reducing log file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        reducing_keys = read_reducing_keys(tmp_dir)
        assert len(reducing_keys) == 0

        write_reducing_key(tmp_dir, "key_1")
        reducing_keys = read_reducing_keys(tmp_dir)
        assert len(reducing_keys) == 1
        assert reducing_keys[0] == "key_1"

        write_reducing_key(tmp_dir, "key_2")
        reducing_keys = read_reducing_keys(tmp_dir)
        assert len(reducing_keys) == 2
        assert reducing_keys[0] == "key_1"
        assert reducing_keys[1] == "key_2"

        clean_resume_files(tmp_dir)
        reducing_keys = read_reducing_keys(tmp_dir)
        assert len(reducing_keys) == 0
