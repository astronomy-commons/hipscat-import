"""Test full execution of SOAP."""

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
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
def test_object_to_source(
    dask_client,
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
):
    """Test creating association between object and source catalogs."""

    args = SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        output_catalog_name="small_sky_association",
        output_path=tmp_path,
        progress_bar=False,
    )
    runner.run(args, dask_client)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_join_pixels()) == 14

    ## Test one pixel that will have 50 rows in it.
    output_file = os.path.join(
        tmp_path,
        "small_sky_association",
        "Norder=0",
        "Dir=0",
        "Npix=11",
        "join_Norder=0",
        "join_Dir=0",
        "join_Npix=4.parquet",
    )
    assert os.path.exists(output_file), f"file not found [{output_file}]"
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["primary_id", "join_id", "join_hipscat_index"],
    )
    assert data_frame.index.name == "primary_hipscat_index"
    assert len(data_frame) == 50
    ids = data_frame["primary_id"]
    assert np.logical_and(ids >= 700, ids < 832).all()
    ids = data_frame["join_id"]
    assert np.logical_and(ids >= 70_000, ids < 87161).all()

    catalog = AssociationCatalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_join_pixels()) == 14
