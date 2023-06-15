"""test stuff."""

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.catalog.association_catalog.association_catalog import AssociationCatalog

import hipscat_import.association.run_association as runner
from hipscat_import.association.arguments import AssociationArguments


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(TypeError):
        runner.run(None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(TypeError):
        runner.run(args)


@pytest.mark.dask
def test_object_to_source(
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
):
    """Test creating association between object and source catalogs."""

    args = AssociationArguments(
        primary_input_catalog_path=small_sky_object_catalog,
        primary_id_column="id",
        primary_join_column="id",
        join_input_catalog_path=small_sky_source_catalog,
        output_catalog_name="small_sky_association",
        join_id_column="source_id",
        join_foreign_key="object_id",
        output_path=tmp_path,
        progress_bar=False,
    )
    runner.run(args)

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


@pytest.mark.dask
def test_source_to_object(
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
):
    """Test creating (weirder) association between source and object catalogs."""

    args = AssociationArguments(
        primary_input_catalog_path=small_sky_source_catalog,
        primary_id_column="source_id",
        primary_join_column="object_id",
        join_input_catalog_path=small_sky_object_catalog,
        join_id_column="id",
        join_foreign_key="id",
        output_catalog_name="small_sky_association",
        output_path=tmp_path,
        progress_bar=False,
    )
    runner.run(args)

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
        "Npix=4",
        "join_Norder=0",
        "join_Dir=0",
        "join_Npix=11.parquet",
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
    assert np.logical_and(ids >= 70_000, ids < 87161).all()
    ids = data_frame["join_id"]
    assert np.logical_and(ids >= 700, ids < 832).all()


@pytest.mark.dask
def test_self_join(
    small_sky_object_catalog,
    tmp_path,
):
    """Test creating association between object catalog and itself."""

    args = AssociationArguments(
        primary_input_catalog_path=small_sky_object_catalog,
        primary_id_column="id",
        primary_join_column="id",
        join_input_catalog_path=small_sky_object_catalog,
        output_catalog_name="small_sky_self_association",
        join_foreign_key="id",
        join_id_column="id",
        output_path=tmp_path,
        progress_bar=False,
    )
    runner.run(args)

    ## Check that the association data can be parsed as a valid association catalog.
    catalog = AssociationCatalog.read_from_hipscat(args.catalog_path)
    assert catalog.on_disk
    assert catalog.catalog_path == args.catalog_path
    assert len(catalog.get_join_pixels()) == 1

    ## Test one pixel that will have 50 rows in it.
    output_file = os.path.join(
        tmp_path,
        "small_sky_self_association",
        "Norder=0",
        "Dir=0",
        "Npix=11",
        "join_Norder=0",
        "join_Dir=0",
        "join_Npix=11.parquet",
    )
    assert os.path.exists(output_file), f"file not found [{output_file}]"
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["primary_id", "join_id", "join_hipscat_index"],
    )
    assert data_frame.index.name == "primary_hipscat_index"
    assert len(data_frame) == 131
    ids = data_frame["primary_id"]
    assert np.logical_and(ids >= 700, ids < 832).all()
    ids = data_frame["join_id"]
    assert np.logical_and(ids >= 700, ids < 832).all()
