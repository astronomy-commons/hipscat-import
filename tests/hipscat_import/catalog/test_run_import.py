"""Functional tests for catalog import"""

import os
import shutil

import pytest
from hipscat.catalog.catalog import Catalog

import hipscat_import.catalog.resume_files as rf
import hipscat_import.catalog.run_import as runner
from hipscat_import.catalog.arguments import ImportArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(ValueError, match="args is required"):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(ValueError, match="ImportArguments"):
        runner.run(args, None)


@pytest.mark.dask
def test_resume_dask_runner(
    dask_client,
    small_sky_parts_dir,
    resume_dir,
    tmp_path,
    assert_parquet_file_ids,
):
    """Test execution in the presence of some resume files."""
    ## First, copy over our intermediate files.
    ## This prevents overwriting source-controlled resume files.
    temp_path = os.path.join(tmp_path, "resume", "intermediate")
    shutil.copytree(
        os.path.join(resume_dir, "intermediate"),
        temp_path,
    )
    for file_index in range(0, 5):
        rf.write_mapping_start_key(
            temp_path,
            f"map_{os.path.join(small_sky_parts_dir, f'catalog_0{file_index}_of_05.csv')}",
        )
        rf.write_mapping_done_key(
            temp_path,
            f'map_{os.path.join(small_sky_parts_dir, f"catalog_0{file_index}_of_05.csv")}',
        )
        rf.write_splitting_done_key(
            temp_path,
            f'split_{os.path.join(small_sky_parts_dir, f"catalog_0{file_index}_of_05.csv")}',
        )

    shutil.copytree(
        os.path.join(resume_dir, "Norder=0"),
        os.path.join(tmp_path, "resume", "Norder=0"),
    )

    with pytest.raises(ValueError, match="resume"):
        ## Check that we fail if there are some existing intermediate files
        ImportArguments(
            output_catalog_name="resume",
            input_path=small_sky_parts_dir,
            input_format="csv",
            output_path=tmp_path,
            dask_tmp=tmp_path,
            tmp_dir=tmp_path,
            overwrite=True,
            highest_healpix_order=0,
            pixel_threshold=1000,
            progress_bar=False,
        )

    args = ImportArguments(
        output_catalog_name="resume",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        overwrite=True,
        resume=True,
        highest_healpix_order=0,
        pixel_threshold=1000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    ## Re-running the pipeline with fully done intermediate files
    ## should result in no changes to output files.
    shutil.copytree(
        os.path.join(resume_dir, "intermediate"),
        temp_path,
    )
    rf.set_mapping_done(temp_path)
    rf.set_splitting_done(temp_path)
    rf.set_reducing_done(temp_path)

    args = ImportArguments(
        output_catalog_name="resume",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        overwrite=True,
        resume=True,
        highest_healpix_order=0,
        pixel_threshold=1000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_pixels()) == 1
    assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.dask
def test_dask_runner(
    dask_client,
    small_sky_parts_dir,
    assert_parquet_file_ids,
    tmp_path,
):
    """Test basic execution."""
    args = ImportArguments(
        output_catalog_name="small_sky_object_catalog",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.dask
def test_dask_runner_stats_only(dask_client, small_sky_parts_dir, tmp_path):
    """Test basic execution, without generating catalog parquet outputs."""
    args = ImportArguments(
        output_catalog_name="small_sky",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
        debug_stats_only=True,
    )

    runner.run(args, dask_client)

    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    assert os.path.exists(metadata_filename)

    # Check that the catalog parquet file DOES NOT exist
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    assert not os.path.exists(output_file)
