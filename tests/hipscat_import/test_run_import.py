"""Test full exection of the dask-parallelized runner"""

import os
import tempfile

import data_paths as dc
import file_testing as ft
import pytest
from dask.distributed import Client, LocalCluster

import hipscat_import.run_import as runner
from hipscat_import.arguments import ImportArguments


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
def test_dask_runner():
    """Test basic execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            highest_healpix_order=1,
            progress_bar=False,
        )

        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            client = Client(cluster)

            runner.run_with_client(args, client)

            # Check that the legacy metadata file exists, and contains correct object data
            expected_lines = [
                "{",
                '    "cat_name": "small_sky",',
                '    "ra_kw": "ra",',
                '    "dec_kw": "dec",',
                '    "id_kw": "id",',
                '    "n_sources": 131,',
                '    "pix_threshold": 1000000,',
                r'    "urls": \[',
                r'        ".*/small_sky_parts/catalog.*.csv"',
                r'        ".*/small_sky_parts/catalog.*.csv"',
                r'        ".*/small_sky_parts/catalog.*.csv"',
                r'        ".*/small_sky_parts/catalog.*.csv"',
                r'        ".*/small_sky_parts/catalog.*.csv"',
                "    ],",
                '    "hips": {',
                r'        "0": \[',
                "            11",
                "        ]",
                "    }",
                "}",
            ]
            metadata_filename = os.path.join(args.catalog_path, "small_sky_meta.json")
            ft.assert_text_file_matches(expected_lines, metadata_filename)

            # Check that the catalog parquet file exists and contains correct object IDs
            output_file = os.path.join(
                args.catalog_path, "Norder0/Npix11", "catalog.parquet"
            )

            expected_ids = [*range(700, 831)]
            ft.assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_dask_runner_stats_only():
    """Test basic execution, without generating catalog parquet outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = ImportArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            dask_tmp=tmp_dir,
            highest_healpix_order=1,
            progress_bar=False,
            debug_stats_only=True,
        )

        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            client = Client(cluster)

            runner.run_with_client(args, client)

            metadata_filename = os.path.join(args.catalog_path, "small_sky_meta.json")
            assert os.path.exists(metadata_filename)

            # Check that the catalog parquet file DOES NOT exist
            output_file = os.path.join(
                args.catalog_path, "Norder0/Npix11", "catalog.parquet"
            )

            assert not os.path.exists(output_file)
