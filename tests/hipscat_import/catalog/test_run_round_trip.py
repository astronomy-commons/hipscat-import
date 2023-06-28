"""Test end-to-end execution of pipeline with different formats and configurations.

Please add a brief description in the docstring of the features or specific 
regression the test case is exercising.
"""


import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.catalog.catalog import Catalog

import hipscat_import.catalog.run_import as runner
from hipscat_import.catalog.arguments import ImportArguments
from hipscat_import.catalog.file_readers import CsvReader, get_file_reader


@pytest.mark.dask
def test_import_source_table(
    dask_client,
    small_sky_source_dir,
    tmp_path,
):
    """Test basic execution, using a larger source file.
    - catalog type should be source
    - will have larger partition info than the corresponding object catalog
    """
    args = ImportArguments(
        output_catalog_name="small_sky_source_catalog",
        input_path=small_sky_source_dir,
        input_format="csv",
        catalog_type="source",
        ra_column="source_ra",
        dec_column="source_dec",
        id_column="source_id",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=2,
        pixel_threshold=3_000,
        overwrite=True,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.ra_column == "source_ra"
    assert catalog.catalog_info.dec_column == "source_dec"
    assert len(catalog.get_pixels()) == 14


@pytest.mark.dask
def test_import_mixed_schema_csv(
    dask_client,
    mixed_schema_csv_dir,
    mixed_schema_csv_parquet,
    assert_parquet_file_ids,
    tmp_path,
):
    """Test basic execution, with a mixed schema.
    - the two input file in `mixed_schema_csv_dir` have different *implied* schemas
        when parsed by pandas. this verifies that they end up with the same schema
        and can be combined into a single parquet file.
    """

    schema_parquet = pd.read_parquet(mixed_schema_csv_parquet)
    args = ImportArguments(
        output_catalog_name="mixed_csv",
        input_path=mixed_schema_csv_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        file_reader=get_file_reader(
            "csv", chunksize=1, type_map=schema_parquet.dtypes.to_dict()
        ),
        progress_bar=False,
        use_schema_file=mixed_schema_csv_parquet,
    )

    runner.run(args, dask_client)

    # Check that the catalog parquet file exists
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    assert_parquet_file_ids(output_file, "id", [*range(700, 708)])


@pytest.mark.dask
def test_import_preserve_index(
    dask_client,
    formats_pandasindex,
    assert_parquet_file_ids,
    assert_parquet_file_index,
    tmp_path,
):
    """Test basic execution, with input with pandas metadata.
    - the input file is a parquet file with some pandas metadata.
        this verifies that the parquet file at the end also has pandas
        metadata, and the user's preferred id is retained as the index,
        when requested.
    """

    expected_indexes = [
        "star1_1",
        "star1_2",
        "star1_3",
        "star1_4",
        "galaxy1_1",
        "galaxy1_2",
        "galaxy2_1",
        "galaxy2_2",
    ]
    assert_parquet_file_index(formats_pandasindex, expected_indexes)
    data_frame = pd.read_parquet(formats_pandasindex, engine="pyarrow")
    assert data_frame.index.name == "obs_id"
    npt.assert_array_equal(
        data_frame.columns,
        ["obj_id", "band", "ra", "dec", "mag"],
    )

    ## Don't generate a hipscat index. Verify that the original index remains.
    args = ImportArguments(
        output_catalog_name="pandasindex",
        input_file_list=[formats_pandasindex],
        input_format="parquet",
        id_column="obs_id",
        add_hipscat_index=False,
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog parquet file exists
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    assert_parquet_file_index(output_file, expected_indexes)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.name == "obs_id"
    npt.assert_array_equal(
        data_frame.columns,
        ["obj_id", "band", "ra", "dec", "mag", "Norder", "Dir", "Npix"],
    )

    ## DO generate a hipscat index. Verify that the original index is preserved in a column.
    args = ImportArguments(
        output_catalog_name="pandasindex_preserve",
        input_file_list=[formats_pandasindex],
        input_format="parquet",
        id_column="obs_id",
        add_hipscat_index=True,
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog parquet file exists
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.name == "_hipscat_index"
    npt.assert_array_equal(
        data_frame.columns,
        ["obs_id", "obj_id", "band", "ra", "dec", "mag", "Norder", "Dir", "Npix"],
    )
    assert_parquet_file_ids(output_file, "obs_id", expected_indexes)


@pytest.mark.dask
def test_import_multiindex(
    dask_client,
    formats_multiindex,
    assert_parquet_file_ids,
    assert_parquet_file_index,
    tmp_path,
):
    """Test basic execution, with input with pandas metadata
    - this is *similar* to the above test
    - the input file is a parquet file with a multi-level pandas index.
        this verifies that the parquet file at the end also has pandas
        metadata, and the user's preferred id is retained as the index,
        when requested.
    """

    index_arrays = [
        [
            "star1",
            "star1",
            "star1",
            "star1",
            "galaxy1",
            "galaxy1",
            "galaxy2",
            "galaxy2",
        ],
        ["r", "r", "i", "i", "r", "r", "r", "r"],
    ]
    expected_indexes = list(zip(index_arrays[0], index_arrays[1]))
    assert_parquet_file_index(formats_multiindex, expected_indexes)
    data_frame = pd.read_parquet(formats_multiindex, engine="pyarrow")
    assert data_frame.index.names == ["obj_id", "band"]
    npt.assert_array_equal(
        data_frame.columns,
        ["ra", "dec", "mag"],
    )

    ## Don't generate a hipscat index. Verify that the original index remains.
    args = ImportArguments(
        output_catalog_name="multiindex",
        input_file_list=[formats_multiindex],
        input_format="parquet",
        id_column=["obj_id", "band"],
        add_hipscat_index=False,
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog parquet file exists
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    assert_parquet_file_index(output_file, expected_indexes)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.names == ["obj_id", "band"]
    npt.assert_array_equal(
        data_frame.columns,
        ["ra", "dec", "mag", "Norder", "Dir", "Npix"],
    )

    ## DO generate a hipscat index. Verify that the original index is preserved in a column.
    args = ImportArguments(
        output_catalog_name="multiindex_preserve",
        input_file_list=[formats_multiindex],
        input_format="parquet",
        id_column=["obj_id", "band"],
        add_hipscat_index=True,
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=1,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog parquet file exists
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.name == "_hipscat_index"
    npt.assert_array_equal(
        data_frame.columns,
        ["obj_id", "band", "ra", "dec", "mag", "Norder", "Dir", "Npix"],
    )
    assert_parquet_file_ids(output_file, "obj_id", index_arrays[0])


@pytest.mark.dask
def test_import_constant_healpix_order(
    dask_client,
    small_sky_parts_dir,
    tmp_path,
):
    """Test basic execution.
    - tests that all the final tiles are at the same healpix order,
        and that we don't create tiles where there is no data.
    """
    args = ImportArguments(
        output_catalog_name="small_sky_object_catalog",
        input_path=small_sky_parts_dir,
        input_format="csv",
        output_path=tmp_path,
        dask_tmp=tmp_path,
        constant_healpix_order=2,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    # Check that the partition info file exists - all pixels at order 2!
    assert all(
        pixel.order == 2 for pixel in catalog.partition_info.get_healpix_pixels()
    )

    # Pick a parquet file and make sure it contains as many rows as we expect
    output_file = os.path.join(
        args.catalog_path, "Norder=2", "Dir=0", "Npix=178.parquet"
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert len(data_frame) == 14
    ids = data_frame["id"]
    assert np.logical_and(ids >= 700, ids < 832).all()

@pytest.mark.dask
def test_import_starr_file(
    dask_client,
    formats_dir,
    assert_parquet_file_ids,
    tmp_path,
):
    """Test basic execution.
    - tests that we can run pipeline with a totally unknown file type, so long
      as a valid InputReader implementation is provided.
    """

    class StarrReader(CsvReader):
        """Shallow subclass"""

    args = ImportArguments(
        output_catalog_name="starr",
        input_path=formats_dir,
        input_format="starr",
        file_reader=StarrReader(),
        output_path=tmp_path,
        dask_tmp=tmp_path,
        highest_healpix_order=2,
        pixel_threshold=3_000,
        progress_bar=False,
    )

    runner.run(args, dask_client)

    # Check that the catalog metadata file exists
    catalog = Catalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert catalog.catalog_info.total_rows == 131
    assert len(catalog.get_pixels()) == 1

    # Check that the catalog parquet file exists and contains correct object IDs
    output_file = os.path.join(
        args.catalog_path, "Norder=0", "Dir=0", "Npix=11.parquet"
    )

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)