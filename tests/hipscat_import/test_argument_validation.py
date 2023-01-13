"""Tests of argument validation, in the absense of command line parsing"""


import tempfile

import pytest

import data_paths as dc
from hipscat_import.arguments import ImportArguments


def test_none():
    """No arguments provided. Should error for required args."""
    args = ImportArguments()
    with pytest.raises(ValueError):
        args.from_params()


def test_empty_required():
    """*Most* required arguments are provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        ## Input path is missing
        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path="",
                input_format="csv",
                output_path=tmp_dir,
            )

        ## Output path is missing
        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path="",
            )


def test_invalid_paths():
    """Required arguments are provided, but paths aren't found."""
    args = ImportArguments()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ## Prove that it works with required args
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            output_path=tmp_dir,
            input_format="csv",
        )

        ## Bad output path
        with pytest.raises(FileNotFoundError):
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                output_path="path",
                input_format="csv",
            )

        ## Bad input path
        with pytest.raises(FileNotFoundError):
            args.from_params(
                catalog_name="catalog",
                input_path="path",
                output_path=tmp_dir,
                input_format="csv",
            )

        ## Input path has no files
        with pytest.raises(FileNotFoundError):
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_EMPTY_DATA_DIR,
                output_path=tmp_dir,
                input_format="csv",
            )

        ## Bad input file
        with pytest.raises(FileNotFoundError):
            args.from_params(
                catalog_name="catalog",
                input_file_list=["path"],
                output_path=tmp_dir,
                input_format="csv",
            )


def test_output_overwrite():
    """Test that we can write to existing directory, but not one with contents"""
    args = ImportArguments()

    with pytest.raises(ValueError):
        args.from_params(
            catalog_name="blank",
            input_path=dc.TEST_BLANK_DATA_DIR,
            output_path=dc.TEST_DATA_DIR,
            input_format="csv",
        )

    ## No error with overwrite flag
    args.from_params(
        catalog_name="blank",
        input_path=dc.TEST_BLANK_DATA_DIR,
        output_path=dc.TEST_DATA_DIR,
        overwrite=True,
        input_format="csv",
    )


def test_good_paths():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            tmp_dir=tmp_dir,
        )
        assert args.input_path == dc.TEST_BLANK_DATA_DIR
        assert len(args.input_paths) == 1
        assert args.input_paths[0] == dc.TEST_BLANK_CSV
        assert args.output_path == tmp_dir
        assert args.tmp_dir.startswith(tmp_dir)


def test_multiple_files_in_path():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
        )
        assert args.input_path == dc.TEST_SMALL_SKY_PARTS_DATA_DIR
        assert len(args.input_paths) == 5


def test_single_debug_file():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="catalog",
            input_file_list=[dc.TEST_FORMATS_HEADERS_CSV],
            input_format="csv",
            output_path=tmp_dir,
        )
        assert len(args.input_paths) == 1
        assert args.input_paths[0] == dc.TEST_FORMATS_HEADERS_CSV


def test_good_paths_empty_args():
    """Paths are good. Remove some required arguments"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
        )
        with pytest.raises(ValueError):
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="",  ## empty
                output_path=tmp_dir,
            )
        with pytest.raises(ValueError):
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path="",  ## empty
            )


def test_dask_args():
    """Test errors for dask arguments"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path=tmp_dir,
                dask_n_workers=-10,
                dask_threads_per_worker=1,
            )

        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path=tmp_dir,
                dask_n_workers=1,
                dask_threads_per_worker=-10,
            )


def test_healpix_args():
    """Test errors for healpix partitioning arguments"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path=tmp_dir,
                highest_healpix_order=30,
            )
        with pytest.raises(ValueError):
            args = ImportArguments()
            args.from_params(
                catalog_name="catalog",
                input_path=dc.TEST_BLANK_DATA_DIR,
                input_format="csv",
                output_path=tmp_dir,
                pixel_threshold=3,
            )


def test_formatted_string():
    """Test that the human readable string contains our specified arguments"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            tmp_dir=tmp_dir,
        )
        formatted_string = str(args)
        assert "catalog" in formatted_string
        assert "csv" in formatted_string
        assert tmp_dir in formatted_string
