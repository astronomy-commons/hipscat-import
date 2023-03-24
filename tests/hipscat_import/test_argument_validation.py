"""Tests of argument validation, in the absense of command line parsing"""


import pytest

from hipscat_import.arguments import ImportArguments

# pylint: disable=protected-access


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        ImportArguments()


def test_empty_required(blank_data_dir, tmp_path):
    """*Most* required arguments are provided."""
    ## Input path is missing
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path="",
            input_format="csv",
            output_path=tmp_path,
        )

    ## Output path is missing
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path="",
        )


def test_invalid_paths(blank_data_dir, empty_data_dir, tmp_path):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    ImportArguments(
        catalog_name="catalog",
        input_path=blank_data_dir,
        output_path=tmp_path,
        input_format="csv",
    )

    ## Bad output path
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            output_path="path",
            input_format="csv",
        )

    ## Bad input path
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            catalog_name="catalog",
            input_path="path",
            output_path=tmp_path,
            overwrite=True,
            input_format="csv",
        )

    ## Input path has no files
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            catalog_name="catalog",
            input_path=empty_data_dir,
            output_path=tmp_path,
            overwrite=True,
            input_format="csv",
        )

    ## Bad input file
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            catalog_name="catalog",
            input_file_list=["path"],
            overwrite=True,
            output_path=tmp_path,
            input_format="csv",
        )


def test_output_overwrite(test_data_dir, blank_data_dir):
    """Test that we can write to existing directory, but not one with contents"""

    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="blank",
            input_path=blank_data_dir,
            output_path=test_data_dir,
            input_format="csv",
        )

    ## No error with overwrite flag
    ImportArguments(
        catalog_name="blank",
        input_path=blank_data_dir,
        output_path=test_data_dir,
        overwrite=True,
        input_format="csv",
    )


def test_good_paths(blank_data_dir, blank_data_file, tmp_path):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = ImportArguments(
        catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path_str,
        tmp_dir=tmp_path_str,
    )
    assert args._input_path == blank_data_dir
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == blank_data_file
    assert args._output_path == tmp_path_str
    assert args._tmp_dir.startswith(tmp_path_str)
    assert args.tmp_path.startswith(tmp_path_str)


def test_multiple_files_in_path(small_sky_parts_dir, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        catalog_name="catalog",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
    )
    assert args._input_path == small_sky_parts_dir
    assert len(args.input_paths) == 5


def test_single_debug_file(formats_headers_csv, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        catalog_name="catalog",
        input_file_list=[formats_headers_csv],
        input_format="csv",
        output_path=tmp_path,
    )
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == formats_headers_csv


def test_good_paths_empty_args(blank_data_dir, tmp_path):
    """Paths are good. Remove some required arguments"""
    ImportArguments(
        catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path,
    )
    with pytest.raises(NotImplementedError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="",  ## empty
            output_path=tmp_path,
        )
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path="",  ## empty
        )


def test_dask_args(blank_data_dir, tmp_path):
    """Test errors for dask arguments"""
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            dask_n_workers=-10,
            dask_threads_per_worker=1,
        )

    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            dask_n_workers=1,
            dask_threads_per_worker=-10,
        )


def test_healpix_args(blank_data_dir, tmp_path):
    """Test errors for healpix partitioning arguments"""
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            highest_healpix_order=30,
        )
    with pytest.raises(ValueError):
        ImportArguments(
            catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            pixel_threshold=3,
        )


def test_to_catalog_parameters(blank_data_dir, tmp_path):
    """Verify creation of catalog parameters for catalog to be created."""
    args = ImportArguments(
        catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
    )
    catalog_parameters = args.to_catalog_parameters()
    assert catalog_parameters.catalog_name == args.catalog_name


def test_formatted_string(blank_data_dir, tmp_path):
    """Test that the human readable string contains our specified arguments"""
    args = ImportArguments(
        catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
    )
    formatted_string = str(args)
    assert "catalog" in formatted_string
    assert "csv" in formatted_string
    assert str(tmp_path) in formatted_string
