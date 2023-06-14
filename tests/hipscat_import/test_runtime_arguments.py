"""Tests of argument validation"""


import os

import pytest

from hipscat_import.runtime_arguments import RuntimeArguments

# pylint: disable=protected-access


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        RuntimeArguments()


def test_empty_required(tmp_path):
    """*Most* required arguments are provided."""
    ## Output path is missing
    with pytest.raises(ValueError, match="output_path"):
        RuntimeArguments(
            output_catalog_name="catalog",
            output_path="",
        )

    ## Output catalog name is missing
    with pytest.raises(ValueError, match="output_catalog_name"):
        RuntimeArguments(
            output_catalog_name="",
            output_path=tmp_path,
        )


def test_catalog_name(tmp_path):
    """Check for safe catalog names."""
    RuntimeArguments(
        output_catalog_name="good_name",
        output_path=tmp_path,
    )

    with pytest.raises(ValueError, match="invalid character"):
        RuntimeArguments(
            output_catalog_name="bad_a$$_name",
            output_path=tmp_path,
        )


def test_invalid_paths(tmp_path):
    """Required arguments are provided, but paths aren't found."""
    ## Bad output path
    with pytest.raises(FileNotFoundError):
        RuntimeArguments(
            output_catalog_name="catalog",
            output_path="/foo/path",
        )

    ## Bad temp path
    with pytest.raises(FileNotFoundError):
        RuntimeArguments(
            output_catalog_name="catalog", output_path=tmp_path, tmp_dir="/foo/path"
        )

    ## Bad dask temp path
    with pytest.raises(FileNotFoundError):
        RuntimeArguments(
            output_catalog_name="catalog", output_path=tmp_path, dask_tmp="/foo/path"
        )


def test_output_overwrite(tmp_path):
    """Test that we can write to existing directory, but not one with contents"""
    ## Create the directory first
    RuntimeArguments(
        output_catalog_name="blank",
        output_path=tmp_path,
    )

    with pytest.raises(ValueError, match="use --overwrite flag"):
        RuntimeArguments(
            output_catalog_name="blank",
            output_path=tmp_path,
        )

    ## No error with overwrite flag
    RuntimeArguments(
        output_catalog_name="blank",
        output_path=tmp_path,
        overwrite=True,
    )


def test_good_paths(tmp_path):
    """Required arguments are provided, and paths are found."""
    _ = RuntimeArguments(
        output_catalog_name="catalog",
        output_path=tmp_path,
        tmp_dir=tmp_path,
        dask_tmp=tmp_path,
    )


def test_tmp_path_creation(tmp_path):
    """Check that we create a new temp path for this catalog."""
    output_path = os.path.join(tmp_path, "unique_output_directory")
    temp_path = os.path.join(tmp_path, "unique_tmp_directory")
    dask_tmp_path = os.path.join(tmp_path, "unique_dask_directory")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(dask_tmp_path, exist_ok=True)

    ## If no tmp paths are given, use the output directory
    args = RuntimeArguments(
        output_catalog_name="special_catalog",
        output_path=output_path,
    )
    assert "special_catalog" in str(args.tmp_path)
    assert "unique_output_directory" in str(args.tmp_path)

    ## Use the tmp path if provided
    args = RuntimeArguments(
        output_catalog_name="special_catalog",
        output_path=output_path,
        tmp_dir=temp_path,
        overwrite=True,
    )
    assert "special_catalog" in str(args.tmp_path)
    assert "unique_tmp_directory" in str(args.tmp_path)

    ## Use the dask tmp for temp, if all else fails
    args = RuntimeArguments(
        output_catalog_name="special_catalog",
        output_path=output_path,
        dask_tmp=dask_tmp_path,
        overwrite=True,
    )
    assert "special_catalog" in str(args.tmp_path)
    assert "unique_dask_directory" in str(args.tmp_path)


def test_dask_args(tmp_path):
    """Test errors for dask arguments"""
    with pytest.raises(ValueError, match="dask_n_workers"):
        RuntimeArguments(
            output_catalog_name="catalog",
            output_path=tmp_path,
            dask_n_workers=-10,
            dask_threads_per_worker=1,
        )

    with pytest.raises(ValueError, match="dask_threads_per_worker"):
        RuntimeArguments(
            output_catalog_name="catalog",
            output_path=tmp_path,
            dask_n_workers=1,
            dask_threads_per_worker=-10,
        )


def test_provenance_info(tmp_path):
    """Verify that provenance info ONLY includes general runtime fields."""
    args = RuntimeArguments(
        output_catalog_name="catalog",
        output_path=tmp_path,
        tmp_dir=tmp_path,
        dask_tmp=tmp_path,
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert len(runtime_args) == 10
