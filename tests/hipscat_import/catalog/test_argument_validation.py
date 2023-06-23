"""Tests of argument validation"""


import pytest

from hipscat_import.catalog.arguments import ImportArguments, check_healpix_order_range

# pylint: disable=protected-access


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        ImportArguments()


def test_empty_required(small_sky_parts_dir, tmp_path):
    """*Most* required arguments are provided."""
    ## Input format is missing
    with pytest.raises(ValueError, match="input_format"):
        ImportArguments(
            output_catalog_name="catalog",
            input_format=None,
            input_path=small_sky_parts_dir,
            output_path=tmp_path,
        )

    ## Input path is missing
    with pytest.raises(ValueError, match="input_file"):
        ImportArguments(
            output_catalog_name="catalog",
            input_path="",
            input_format="csv",
            output_path=tmp_path,
            overwrite=True,
        )


def test_invalid_paths(blank_data_dir, tmp_path):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    ImportArguments(
        output_catalog_name="catalog",
        input_path=blank_data_dir,
        output_path=tmp_path,
        input_format="csv",
    )

    ## Bad input path
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            output_catalog_name="catalog",
            input_path="path",
            output_path=tmp_path,
            overwrite=True,
            input_format="csv",
        )

    ## Input path has no files
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            output_catalog_name="catalog",
            input_path=blank_data_dir,
            output_path=tmp_path,
            overwrite=True,
            input_format="parquet",
        )

    ## Bad input file
    with pytest.raises(FileNotFoundError):
        ImportArguments(
            output_catalog_name="catalog",
            input_file_list=["/foo/path"],
            overwrite=True,
            output_path=tmp_path,
            input_format="csv",
        )


def test_good_paths(blank_data_dir, blank_data_file, tmp_path):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = ImportArguments(
        output_catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path_str,
        tmp_dir=tmp_path_str,
    )
    assert args.input_path == blank_data_dir
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == blank_data_file


def test_multiple_files_in_path(small_sky_parts_dir, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        output_catalog_name="catalog",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
    )
    assert args.input_path == small_sky_parts_dir
    assert len(args.input_paths) == 5


def test_single_debug_file(formats_headers_csv, tmp_path):
    """Required arguments are provided, and paths are found."""
    args = ImportArguments(
        output_catalog_name="catalog",
        input_file_list=[formats_headers_csv],
        input_format="csv",
        output_path=tmp_path,
    )
    assert len(args.input_paths) == 1
    assert args.input_paths[0] == formats_headers_csv


def test_healpix_args(blank_data_dir, tmp_path):
    """Test errors for healpix partitioning arguments"""
    with pytest.raises(ValueError, match="highest_healpix_order"):
        ImportArguments(
            output_catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            highest_healpix_order=30,
            overwrite=True,
        )
    with pytest.raises(ValueError, match="pixel_threshold"):
        ImportArguments(
            output_catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            pixel_threshold=3,
            overwrite=True,
        )
    with pytest.raises(ValueError, match="constant_healpix_order"):
        ImportArguments(
            output_catalog_name="catalog",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
            constant_healpix_order=30,
            overwrite=True,
        )


def test_catalog_type(blank_data_dir, tmp_path):
    """Test errors for catalog types."""
    with pytest.raises(ValueError, match="catalog_type"):
        ImportArguments(
            output_catalog_name="catalog",
            catalog_type=None,
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
        )

    with pytest.raises(ValueError, match="catalog_type"):
        ImportArguments(
            output_catalog_name="catalog",
            catalog_type="association",
            input_path=blank_data_dir,
            input_format="csv",
            output_path=tmp_path,
        )


def test_to_catalog_info(blank_data_dir, tmp_path):
    """Verify creation of catalog parameters for catalog to be created."""
    args = ImportArguments(
        output_catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
    )
    catalog_info = args.to_catalog_info(total_rows=10)
    assert catalog_info.catalog_name == "catalog"
    assert catalog_info.total_rows == 10


def test_provenance_info(blank_data_dir, tmp_path):
    """Verify that provenance info includes association-specific fields."""
    args = ImportArguments(
        output_catalog_name="catalog",
        input_path=blank_data_dir,
        input_format="csv",
        output_path=tmp_path,
        tmp_dir=tmp_path,
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "epoch" in runtime_args


def test_check_healpix_order_range():
    """Test method check_healpix_order_range"""
    check_healpix_order_range(5, "order_field")
    check_healpix_order_range(5, "order_field", lower_bound=0, upper_bound=19)

    with pytest.raises(ValueError, match="positive"):
        check_healpix_order_range(5, "order_field", lower_bound=-1)

    with pytest.raises(ValueError, match="19"):
        check_healpix_order_range(5, "order_field", upper_bound=20)

    with pytest.raises(ValueError, match="order_field"):
        check_healpix_order_range(-1, "order_field")
    with pytest.raises(ValueError, match="order_field"):
        check_healpix_order_range(30, "order_field")

    with pytest.raises(TypeError, match="not supported"):
        check_healpix_order_range("two", "order_field")
    with pytest.raises(TypeError, match="not supported"):
        check_healpix_order_range(5, "order_field", upper_bound="ten")
