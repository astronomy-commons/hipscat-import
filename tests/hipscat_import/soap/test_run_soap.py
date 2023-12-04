"""Test full execution of SOAP."""
import pytest
from hipscat.catalog.association_catalog.association_catalog import AssociationCatalog

import hipscat_import.soap.run_soap as runner
from hipscat_import.soap.arguments import SoapArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError):
        runner.run(None, None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(TypeError):
        runner.run(args, None)


@pytest.mark.dask
def test_object_to_source(dask_client, small_sky_soap_args):
    """Test creating association between object and source catalogs."""
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
    assert catalog.catalog_info.total_rows == 17161


@pytest.mark.dask
def test_object_to_source_with_leaves(
    dask_client, tmp_path, small_sky_object_catalog, small_sky_source_catalog
):
    """Test creating association between object and source catalogs."""
    small_sky_soap_args = SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        output_path=tmp_path,
        overwrite=True,
        progress_bar=False,
        write_leaf_files=True,
        source_id_column="source_id",
        output_artifact_name="small_sky_object_to_source",
    )
    runner.run(small_sky_soap_args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(small_sky_soap_args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == small_sky_soap_args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
    assert catalog.catalog_info.total_rows == 17161
