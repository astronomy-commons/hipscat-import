"""Test full exection of the dask-parallelized runner"""

import os
import shutil
import tempfile

import file_testing as ft
import pandas as pd
import pytest

import hipscat_import.resume_files as rf
import hipscat_import.run_import as runner
from hipscat_import.arguments import ImportArguments
from hipscat_import.file_readers import get_file_reader


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(ValueError):
        runner.run(None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"catalog_name": "bad_arg_type"}
    with pytest.raises(ValueError):
        runner.run(args)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_resume_dask_runner(dask_client, small_sky_parts_dir, resume_dir):
    """Test execution in the presence of some resume files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ## First, copy over our intermediate files.
        ## This prevents overwriting source-controlled resume files.
        temp_path = os.path.join(tmp_dir, "resume", "intermediate")
        shutil.copytree(
            os.path.join(resume_dir, "intermediate"),
            temp_path,
        )

        shutil.copytree(
            os.path.join(resume_dir, "Norder0"),
            os.path.join(tmp_dir, "resume", "Norder0"),
        )

        with pytest.raises(ValueError):
            ## Check that we fail if there are some existing intermediate files
            ImportArguments(
                catalog_name="resume",
                input_path=small_sky_parts_dir,
                input_format="csv",
                output_path=tmp_dir,
                dask_tmp=tmp_dir,
                tmp_dir=tmp_dir,
                overwrite=True,
                highest_healpix_order=0,
                pixel_threshold=1000,
                progress_bar=False,
            )

        args = ImportArguments(
            catalog_name="resume",
            input_path=small_sky_parts_dir,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            tmp_dir=tmp_dir,
            overwrite=True,
            resume=True,
            highest_healpix_order=0,
            pixel_threshold=1000,
            progress_bar=False,
        )

        runner.run_with_client(args, dask_client)

        # Check that the catalog metadata file exists
        expected_metadata_lines = [
            "{",
            '    "catalog_name": "resume",',
            r'    "version": "[.\d]+.*",',  # version matches digits
            r'    "generation_date": "[.\d]+",',  # date matches date format
            '    "ra_kw": "ra",',
            '    "dec_kw": "dec",',
            '    "id_kw": "id",',
            '    "total_objects": 131,',
            '    "origin_healpix_order": 0',
            '    "pixel_threshold": 1000',
            "}",
        ]
        metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
        ft.assert_text_file_matches(expected_metadata_lines, metadata_filename)

        # Check that the partition info file exists
        expected_partition_lines = [
            "order,pixel,num_objects",
            "0,11,131",
        ]
        metadata_filename = os.path.join(args.catalog_path, "partition_info.csv")
        ft.assert_text_file_matches(expected_partition_lines, metadata_filename)

        # Check that the catalog parquet file exists and contains correct object IDs
        output_file = os.path.join(
            args.catalog_path, "Norder0/Npix11", "catalog.parquet"
        )

        expected_ids = [*range(700, 831)]
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)

        ## Re-running the pipeline with fully done intermediate files
        ## should result in no changes to output files.
        shutil.copytree(
            os.path.join(resume_dir, "intermediate"),
            temp_path,
        )
        rf.set_mapping_done(temp_path)
        rf.set_reducing_done(temp_path)

        args = ImportArguments(
            catalog_name="resume",
            input_path=small_sky_parts_dir,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            tmp_dir=tmp_dir,
            overwrite=True,
            resume=True,
            highest_healpix_order=0,
            pixel_threshold=1000,
            progress_bar=False,
        )

        runner.run_with_client(args, dask_client)

        ft.assert_text_file_matches(expected_metadata_lines, metadata_filename)
        ft.assert_text_file_matches(expected_partition_lines, metadata_filename)
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_dask_runner(dask_client, small_sky_parts_dir):
    """Test basic execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments(
            catalog_name="small_sky",
            input_path=small_sky_parts_dir,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            highest_healpix_order=1,
            progress_bar=False,
        )

        runner.run_with_client(args, dask_client)

        # Check that the catalog metadata file exists
        expected_lines = [
            "{",
            '    "catalog_name": "small_sky",',
            r'    "version": "[.\d]+.*",',  # version matches digits
            r'    "generation_date": "[.\d]+",',  # date matches date format
            '    "ra_kw": "ra",',
            '    "dec_kw": "dec",',
            '    "id_kw": "id",',
            '    "total_objects": 131,',
            '    "origin_healpix_order": 1',
            '    "pixel_threshold": 1000000',
            "}",
        ]
        metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
        ft.assert_text_file_matches(expected_lines, metadata_filename)

        # Check that the partition info file exists
        expected_lines = [
            "order,pixel,num_objects",
            "0,11,131",
        ]
        metadata_filename = os.path.join(args.catalog_path, "partition_info.csv")
        ft.assert_text_file_matches(expected_lines, metadata_filename)

        # Check that the catalog parquet file exists and contains correct object IDs
        output_file = os.path.join(
            args.catalog_path, "Norder0/Npix11", "catalog.parquet"
        )

        expected_ids = [*range(700, 831)]
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


def test_dask_runner_stats_only(dask_client, small_sky_parts_dir):
    """Test basic execution, without generating catalog parquet outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments(
            catalog_name="small_sky",
            input_path=small_sky_parts_dir,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            highest_healpix_order=1,
            progress_bar=False,
            debug_stats_only=True,
        )

        runner.run_with_client(args, dask_client)

        metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
        assert os.path.exists(metadata_filename)

        # Check that the catalog parquet file DOES NOT exist
        output_file = os.path.join(
            args.catalog_path, "Norder0/Npix11", "catalog.parquet"
        )

        assert not os.path.exists(output_file)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_dask_runner_mixed_schema_csv(
    dask_client, mixed_schema_csv_dir, mixed_schema_csv_parquet
):
    """Test basic execution, with a mixed schema"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        schema_parquet = pd.read_parquet(mixed_schema_csv_parquet)
        args = ImportArguments(
            catalog_name="mixed_csv",
            input_path=mixed_schema_csv_dir,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            highest_healpix_order=1,
            file_reader=get_file_reader(
                "csv", chunksize=1, type_map=schema_parquet.dtypes.to_dict()
            ),
            progress_bar=False,
        )

        runner.run_with_client(args, dask_client)

        # Check that the catalog parquet file exists
        output_file = os.path.join(
            args.catalog_path, "Norder0/Npix11", "catalog.parquet"
        )

        ft.assert_parquet_file_ids(output_file, "id", [*range(700, 708)])
