"""Tests that create a new primary object catalog, and immediately 
create a margin cache based on the primary catalog."""

import os

import pandas as pd
import pytest
from hipscat.catalog.catalog import Catalog
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset
from hipscat.io import paths

import hipscat_import.catalog.run_import as runner
import hipscat_import.margin_cache.margin_cache as mc
from hipscat_import.catalog.arguments import ImportArguments
from hipscat_import.catalog.file_readers import CsvReader, get_file_reader
from hipscat_import.margin_cache.margin_cache_arguments import MarginCacheArguments


@pytest.mark.dask(timeout=180)
def test_margin_import_gaia_minimum(
    dask_client,
    formats_dir,
    tmp_path,
):
    """Create a very small representative gaia catalog, and create a margin catalog
    using a very large margin_threshold."""
    input_file = os.path.join(formats_dir, "gaia_minimum.csv")
    schema_file = os.path.join(formats_dir, "gaia_minimum_schema.parquet")

    args = ImportArguments(
        output_artifact_name="gaia_minimum",
        input_file_list=[input_file],
        file_reader=CsvReader(
            comment="#",
            schema_file=schema_file,
        ),
        ra_column="ra",
        dec_column="dec",
        sort_columns="solution_id",
        use_schema_file=schema_file,
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=2,
        pixel_threshold=3_000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    Catalog.read_from_hipscat(args.catalog_path)

    args = MarginCacheArguments(
        margin_threshold=180.0,
        input_catalog_path=args.catalog_path,
        output_path=tmp_path,
        output_artifact_name="gaia_10arcsec",
        margin_order=8,
        progress_bar=False,
    )

    mc.generate_margin_cache(args, dask_client)
    catalog = HealpixDataset.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 1

    norder = 0
    npix = 0

    test_file = paths.pixel_catalog_file(args.catalog_path, norder, npix)

    data = pd.read_parquet(test_file)

    assert len(data) == 1


@pytest.mark.dask(timeout=180)
def test_margin_import_mixed_schema_csv(
    dask_client,
    mixed_schema_csv_dir,
    mixed_schema_csv_parquet,
    tmp_path,
):
    """Test basic execution, with a mixed schema.
    - the two input files in `mixed_schema_csv_dir` have different *implied* schemas
        when parsed by pandas. this verifies that they end up with the same schema
        and can be combined into a single parquet file.
    """
    args = ImportArguments(
        output_artifact_name="mixed_csv",
        input_file_list=[
            os.path.join(mixed_schema_csv_dir, "input_01.csv"),
            os.path.join(mixed_schema_csv_dir, "input_02.csv"),
        ],
        output_path=tmp_path,
        dask_tmp=tmp_path,
        constant_healpix_order=3,
        file_reader=get_file_reader("csv", chunksize=1, schema_file=mixed_schema_csv_parquet),
        use_schema_file=mixed_schema_csv_parquet,
        progress_bar=False,
    )
    runner.run(args, dask_client)
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert len(catalog.get_healpix_pixels()) == 8

    args = MarginCacheArguments(
        margin_threshold=3600.0,
        input_catalog_path=args.catalog_path,
        output_path=tmp_path,
        output_artifact_name="mixed_csv_36000arcsec",
        margin_order=4,
        progress_bar=False,
    )

    mc.generate_margin_cache(args, dask_client)
    catalog = HealpixDataset.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_healpix_pixels()) == 5

    norder = 2
    npix = 187

    test_file = paths.pixel_catalog_file(args.catalog_path, norder, npix)

    data = pd.read_parquet(test_file)

    assert len(data) == 1
