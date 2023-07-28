"""Test full execution of SOAP."""
import pytest
from hipscat.catalog.association_catalog.association_catalog import AssociationCatalog

import hipscat_import.soap.run_soap as runner


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
