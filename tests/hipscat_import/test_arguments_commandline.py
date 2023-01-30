"""Tests of command line argument validation"""

import tempfile

import pytest

import data_paths as dc
from hipscat_import.arguments import ImportArguments


def test_none():
    """No arguments provided. Should error for required args."""
    empty_args = []
    args = ImportArguments()
    with pytest.raises(ValueError):
        args.from_command_line(empty_args)


def test_invalid_arguments():
    """Arguments are ill-formed."""
    bad_form_args = ["catalog", "path", "path"]
    args = ImportArguments()
    with pytest.raises(SystemExit):
        args.from_command_line(bad_form_args)


def test_invalid_path():
    """Required arguments are provided, but paths aren't found."""
    bad_path_args = ["-c", "catalog", "-i", "path", "-o", "path"]
    args = ImportArguments()
    with pytest.raises(FileNotFoundError):
        args.from_command_line(bad_path_args)


def test_good_paths():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        good_args = [
            "--catalog_name",
            "catalog",
            "--input_path",
            dc.TEST_BLANK_DATA_DIR,
            "--output_path",
            tmp_dir,
            "--input_format",
            "csv",
        ]
        args = ImportArguments()
        args.from_command_line(good_args)
        assert args.input_path == dc.TEST_BLANK_DATA_DIR
        assert args.output_path == tmp_dir


def test_good_paths_short_names():
    """Required arguments are provided, using short names for arguments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        good_args = [
            "-c",
            "catalog",
            "-i",
            dc.TEST_BLANK_DATA_DIR,
            "-o",
            tmp_dir,
            "-fmt",
            "csv",
        ]
        args = ImportArguments()
        args.from_command_line(good_args)
        assert args.input_path == dc.TEST_BLANK_DATA_DIR
        assert args.output_path == tmp_dir
