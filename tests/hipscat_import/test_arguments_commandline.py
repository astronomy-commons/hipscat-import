"""Tests of command line argument validation"""

import tempfile

import pytest

from hipscat_import.command_line_arguments import parse_command_line


def test_none():
    """No arguments provided. Should error for required args."""
    empty_args = []
    with pytest.raises(ValueError):
        parse_command_line(empty_args)


def test_invalid_arguments():
    """Arguments are ill-formed."""
    bad_form_args = ["catalog", "path", "path"]
    with pytest.raises(SystemExit):
        parse_command_line(bad_form_args)


def test_invalid_path():
    """Required arguments are provided, but paths aren't found."""
    bad_path_args = ["-c", "catalog", "-i", "path", "-o", "path"]
    with pytest.raises(FileNotFoundError):
        parse_command_line(bad_path_args)


def test_good_paths(blank_data_dir):
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        good_args = [
            "--catalog_name",
            "catalog",
            "--input_path",
            blank_data_dir,
            "--output_path",
            tmp_dir,
            "--input_format",
            "csv",
        ]
        args = parse_command_line(good_args)
        assert args.input_path == blank_data_dir
        assert args.output_path == tmp_dir


def test_good_paths_short_names(blank_data_dir):
    """Required arguments are provided, using short names for arguments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        good_args = [
            "-c",
            "catalog",
            "-i",
            blank_data_dir,
            "-o",
            tmp_dir,
            "-fmt",
            "csv",
        ]
        args = parse_command_line(good_args)
        assert args.input_path == blank_data_dir
        assert args.output_path == tmp_dir
