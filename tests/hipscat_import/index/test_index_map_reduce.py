"""Tests of map reduce operations"""


import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import hipscat_import.index.map_reduce as mr
from hipscat_import.index.arguments import IndexArguments


@pytest.mark.dask
def test_create_index(
    small_sky_object_catalog,
    assert_parquet_file_index,
    tmp_path,
):
    """Create an index for simple object catalog"""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        output_path=tmp_path,
        output_catalog_name="small_sky_object_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_object_index", "index", "part.0.parquet"
    )

    expected_ids = [*range(700, 831)]
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_hipscat_index", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "id"
    assert (data_frame["Norder"] == 0).all()


@pytest.mark.dask
def test_create_index_no_hipscat_index(small_sky_object_catalog, tmp_path):
    """Create an index for simple object catalog"""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        include_hipscat_index=False,
        output_path=tmp_path,
        output_catalog_name="small_sky_object_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_object_index", "index", "part.0.parquet"
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(data_frame.columns, ["Norder", "Dir", "Npix"])
    assert data_frame.index.name == "id"


@pytest.mark.dask
def test_create_index_no_order_pixel(small_sky_object_catalog, tmp_path):
    """Create an index for simple object catalog"""
    args = IndexArguments(
        input_catalog_path=small_sky_object_catalog,
        indexing_column="id",
        include_order_pixel=False,
        output_path=tmp_path,
        output_catalog_name="small_sky_object_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_object_index", "index", "part.0.parquet"
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(data_frame.columns, ["_hipscat_index"])
    assert data_frame.index.name == "id"


@pytest.mark.dask
def test_create_index_source(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
):
    """test stuff"""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="source_id",
        output_path=tmp_path,
        output_catalog_name="small_sky_source_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_source_index", "index", "part.0.parquet"
    )

    expected_ids = [*range(70_000, 87_161)]
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_hipscat_index", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "source_id"
    assert len(data_frame) == 17161
    assert np.logical_and(data_frame["Norder"] >= 0, data_frame["Norder"] <= 2).all()


@pytest.mark.dask
def test_create_index_source_by_object(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
):
    """test stuff"""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="object_id",
        output_path=tmp_path,
        output_catalog_name="small_sky_source_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_source_index", "index", "part.0.parquet"
    )

    expected_ids = np.repeat([*range(700, 831)], 131)
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_hipscat_index", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "object_id"
    assert len(data_frame) == 17161


@pytest.mark.dask
def test_create_index_extra_columns(
    small_sky_source_catalog,
    assert_parquet_file_index,
    tmp_path,
):
    """test stuff"""
    args = IndexArguments(
        input_catalog_path=small_sky_source_catalog,
        indexing_column="object_id",
        output_path=tmp_path,
        extra_columns=["source_ra"],
        output_catalog_name="small_sky_source_index",
        overwrite=True,
        progress_bar=False,
    )
    mr.create_index(args)

    output_file = os.path.join(
        tmp_path, "small_sky_source_index", "index", "part.0.parquet"
    )

    expected_ids = np.repeat([*range(700, 831)], 131)
    assert_parquet_file_index(output_file, expected_ids)

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    npt.assert_array_equal(
        data_frame.columns,
        ["_hipscat_index", "source_ra", "Norder", "Dir", "Npix"],
    )
    assert data_frame.index.name == "object_id"
    assert len(data_frame) == 17161
