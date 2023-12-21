import os

import pytest
from hipscat.catalog.association_catalog.association_catalog import AssociationCatalog
from hipscat.io import file_io

import hipscat_import.cross_match.run_macauff_import as runner
from hipscat_import.catalog.file_readers import CsvReader
from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat_import.cross_match.macauff_metadata import from_yaml

# pylint: disable=too-many-instance-attributes
# pylint: disable=duplicate-code


@pytest.mark.dask
def test_bad_args(dask_client):
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="MacauffArguments"):
        runner.run(None, dask_client)

    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="MacauffArguments"):
        runner.run(args, dask_client)


@pytest.mark.dask
def test_object_to_object(
    small_sky_object_catalog,
    tmp_path,
    macauff_data_dir,
    dask_client,
):
    """Test that we can create a MacauffArguments instance with two valid catalogs."""

    yaml_input_file = os.path.join(macauff_data_dir, "macauff_gaia_catwise_match_and_nonmatches.yaml")
    from_yaml(yaml_input_file, tmp_path)
    matches_schema_file = os.path.join(tmp_path, "macauff_GaiaDR3xCatWISE2020_matches.parquet")
    single_metadata = file_io.read_parquet_metadata(matches_schema_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 7

    args = MacauffArguments(
        output_path=tmp_path,
        output_artifact_name="object_to_object",
        tmp_dir=tmp_path,
        left_catalog_dir=small_sky_object_catalog,
        left_ra_column="gaia_ra",
        left_dec_column="gaia_dec",
        left_id_column="gaia_source_id",
        right_catalog_dir=small_sky_object_catalog,
        right_ra_column="catwise_ra",
        right_dec_column="catwise_dec",
        right_id_column="catwise_name",
        input_file_list=[os.path.join(macauff_data_dir, "gaia_small_sky_matches.csv")],
        input_format="csv",
        overwrite=True,
        file_reader=CsvReader(schema_file=matches_schema_file, header=None),
        metadata_file_path=matches_schema_file,
        # progress_bar=False,
    )
    os.makedirs(os.path.join(args.tmp_path, "splitting"))

    runner.run(args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_join_pixels()) == 1
    assert catalog.catalog_info.total_rows == 131


@pytest.mark.dask
def test_source_to_object(
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
    macauff_data_dir,
    dask_client,
):
    """Test that we can create a MacauffArguments instance with two valid catalogs."""

    yaml_input_file = os.path.join(macauff_data_dir, "macauff_gaia_catwise_match_and_nonmatches.yaml")
    from_yaml(yaml_input_file, tmp_path)
    matches_schema_file = os.path.join(tmp_path, "macauff_GaiaDR3xCatWISE2020_matches.parquet")
    single_metadata = file_io.read_parquet_metadata(matches_schema_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 7

    args = MacauffArguments(
        output_path=tmp_path,
        output_artifact_name="object_to_object",
        tmp_dir=tmp_path,
        left_catalog_dir=small_sky_source_catalog,
        left_ra_column="gaia_ra",
        left_dec_column="gaia_dec",
        left_id_column="gaia_source_id",
        right_catalog_dir=small_sky_object_catalog,
        right_ra_column="catwise_ra",
        right_dec_column="catwise_dec",
        right_id_column="catwise_name",
        input_file_list=[os.path.join(macauff_data_dir, "small_sky_and_source_matches.csv")],
        input_format="csv",
        overwrite=True,
        file_reader=CsvReader(schema_file=matches_schema_file, header=None),
        metadata_file_path=matches_schema_file,
        progress_bar=False,
    )
    os.makedirs(os.path.join(args.tmp_path, "splitting"))

    runner.run(args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_join_pixels()) == 8
    assert catalog.catalog_info.total_rows == 34
