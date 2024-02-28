"""Functional tests for catalog import"""

import os
import shutil

import hipscat.pixel_math as hist
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hipscat.catalog.catalog import Catalog

import hipscat_import.catalog.run_import as runner
from hipscat_import.catalog.arguments import ImportArguments
from hipscat_import.catalog.file_readers import CsvReader
from hipscat_import.catalog.resume_plan import ResumePlan


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(ValueError, match="args is required"):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_artifact_name": "bad_arg_type"}
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
    intermediate_dir = os.path.join(tmp_path, "resume_catalog", "intermediate")
    shutil.copytree(
        os.path.join(resume_dir, "intermediate"),
        intermediate_dir,
    )
    ## Now set up our resume files to match previous work.
    resume_tmp = os.path.join(tmp_path, "tmp", "resume_catalog")
    plan = ResumePlan(tmp_path=resume_tmp, progress_bar=False)
    histogram = hist.empty_histogram(0)
    histogram[11] = 131
    empty = hist.empty_histogram(0)
    for file_index in range(0, 5):
        ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.SPLITTING_STAGE, f"split_{file_index}")
        ResumePlan.write_partial_histogram(
            tmp_path=resume_tmp,
            mapping_key=f"map_{file_index}",
            histogram=histogram if file_index == 0 else empty,
        )

    ResumePlan.touch_key_done_file(resume_tmp, ResumePlan.REDUCING_STAGE, "0_11")

    shutil.copytree(
        os.path.join(resume_dir, "Norder=0"),
        os.path.join(tmp_path, "resume_catalog", "Norder=0"),
    )

    with pytest.warns(UserWarning, match="resuming prior progress"):
        args = ImportArguments(
            output_artifact_name="resume_catalog",
            input_path=small_sky_parts_dir,
            file_reader="csv",
            output_path=tmp_path,
            dask_tmp=tmp_path,
            tmp_dir=tmp_path,
            resume_tmp=os.path.join(tmp_path, "tmp"),
            overwrite=True,
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
    assert len(catalog.get_healpix_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    ## Re-running the pipeline with fully done intermediate files
    ## should result in no changes to output files.
    shutil.copytree(
        os.path.join(resume_dir, "intermediate"),
        resume_tmp,
    )
    plan = args.resume_plan
    plan.touch_stage_done_file(ResumePlan.MAPPING_STAGE)
    plan.touch_stage_done_file(ResumePlan.SPLITTING_STAGE)
    plan.touch_stage_done_file(ResumePlan.REDUCING_STAGE)

    args = ImportArguments(
        output_artifact_name="resume",
        input_path=small_sky_parts_dir,
        file_reader="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        tmp_dir=tmp_path,
        overwrite=True,
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
    assert len(catalog.get_healpix_pixels()) == 1
    assert_parquet_file_ids(output_file, "id", expected_ids)


@pytest.mark.dask
def test_dask_runner(
    dask_client,
    small_sky_parts_dir,
    assert_parquet_file_ids,
    tmp_path,
):
    """Test basic execution and the types of the written data."""
    args = ImportArguments(
        output_artifact_name="small_sky_object_catalog",
        input_path=small_sky_parts_dir,
        file_reader=CsvReader(
            type_map={
                "ra": np.float32,
                "dec": np.float32,
                "ra_error": np.float32,
                "dec_error": np.float32,
            }
        ),
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
    assert len(catalog.get_healpix_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)

    # Check that the schema is correct for leaf parquet and _metadata files
    expected_parquet_schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("ra", pa.float32()),
            pa.field("dec", pa.float32()),
            pa.field("ra_error", pa.float32()),
            pa.field("dec_error", pa.float32()),
            pa.field("Norder", pa.uint8()),
            pa.field("Dir", pa.uint64()),
            pa.field("Npix", pa.uint64()),
            pa.field("_hipscat_index", pa.uint64()),
        ]
    )
    schema = pq.read_metadata(output_file).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema, check_metadata=False)
    schema = pq.read_metadata(os.path.join(args.catalog_path, "_metadata")).schema.to_arrow_schema()
    assert schema.equals(expected_parquet_schema, check_metadata=False)

    # Check that, when re-loaded as a pandas dataframe, the appropriate numeric types are used.
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dtypes = pd.Series(
        {
            "id": np.int64,
            "ra": np.float32,
            "dec": np.float32,
            "ra_error": np.float32,
            "dec_error": np.float32,
            "Norder": np.uint8,
            "Dir": np.uint64,
            "Npix": np.uint64,
        }
    )
    assert data_frame.dtypes.equals(expected_dtypes)
    assert data_frame.index.dtype == np.uint64


@pytest.mark.dask
def test_dask_runner_stats_only(dask_client, small_sky_parts_dir, tmp_path):
    """Test basic execution, without generating catalog parquet outputs."""
    args = ImportArguments(
        output_artifact_name="small_sky",
        input_path=small_sky_parts_dir,
        file_reader="csv",
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
    output_file = os.path.join(args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    assert not os.path.exists(output_file)

    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "ra"
    assert catalog.catalog_info.dec_column == "dec"
    assert len(catalog.get_healpix_pixels()) == 1
