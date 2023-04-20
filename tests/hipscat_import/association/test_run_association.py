"""test stuff."""

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.timeout(15)
def test_object_to_source(
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
    assert_text_file_matches,
):
    """test stuff"""

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

    # Check that the catalog metadata file exists
    expected_metadata_lines = [
        "{",
        '    "catalog_name": "small_sky_association",',
        '    "catalog_type": "association",',
        '    "epoch": "J2000",',
        '    "ra_kw": "ra",',
        '    "dec_kw": "dec",',
        '    "total_rows": 17161',
        "}",
    ]
    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    assert_text_file_matches(expected_metadata_lines, metadata_filename)

    # Check that the partition *join* info file exists
    expected_lines = [
        "Norder,Dir,Npix,join_Norder,join_Dir,join_Npix,num_rows",
        "0,0,11,0,0,4,50",
        "0,0,11,1,0,47,2395",
        "0,0,11,2,0,176,385",
        "0,0,11,2,0,177,1510",
        "0,0,11,2,0,178,1634",
        "0,0,11,2,0,179,1773",
        "0,0,11,2,0,180,655",
        "0,0,11,2,0,181,903",
        "0,0,11,2,0,182,1246",
        "0,0,11,2,0,183,1143",
        "0,0,11,2,0,184,1390",
        "0,0,11,2,0,185,2942",
        "0,0,11,2,0,186,452",
        "0,0,11,2,0,187,683",
    ]
    metadata_filename = os.path.join(args.catalog_path, "partition_join_info.csv")
    assert_text_file_matches(expected_lines, metadata_filename)

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.timeout(15)
def test_source_to_object(
    small_sky_object_catalog,
    small_sky_source_catalog,
    tmp_path,
    assert_text_file_matches,
):
    """test stuff"""

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

    # Check that the catalog metadata file exists
    expected_metadata_lines = [
        "{",
        '    "catalog_name": "small_sky_association",',
        '    "catalog_type": "association",',
        '    "epoch": "J2000",',
        '    "ra_kw": "ra",',
        '    "dec_kw": "dec",',
        '    "total_rows": 17161',
        "}",
    ]
    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    assert_text_file_matches(expected_metadata_lines, metadata_filename)

    # Check that the partition *join* info file exists
    expected_lines = [
        "Norder,Dir,Npix,join_Norder,join_Dir,join_Npix,num_rows",
        "0,0,4,0,0,11,50",
        "1,0,47,0,0,11,2395",
        "2,0,176,0,0,11,385",
        "2,0,177,0,0,11,1510",
        "2,0,178,0,0,11,1634",
        "2,0,179,0,0,11,1773",
        "2,0,180,0,0,11,655",
        "2,0,181,0,0,11,903",
        "2,0,182,0,0,11,1246",
        "2,0,183,0,0,11,1143",
        "2,0,184,0,0,11,1390",
        "2,0,185,0,0,11,2942",
        "2,0,186,0,0,11,452",
        "2,0,187,0,0,11,683",
    ]
    metadata_filename = os.path.join(args.catalog_path, "partition_join_info.csv")
    assert_text_file_matches(expected_lines, metadata_filename)

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.timeout(15)
def test_self_join(
    small_sky_object_catalog,
    tmp_path,
    assert_text_file_matches,
):
    """test stuff"""

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

    # Check that the catalog metadata file exists
    expected_metadata_lines = [
        "{",
        '    "catalog_name": "small_sky_self_association",',
        '    "catalog_type": "association",',
        '    "epoch": "J2000",',
        '    "ra_kw": "ra",',
        '    "dec_kw": "dec",',
        '    "total_rows": 131',
        "}",
    ]
    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    assert_text_file_matches(expected_metadata_lines, metadata_filename)

    # Check that the partition *join* info file exists
    expected_lines = [
        "Norder,Dir,Npix,join_Norder,join_Dir,join_Npix,num_rows",
        "0,0,11,0,0,11,131",
    ]
    metadata_filename = os.path.join(args.catalog_path, "partition_join_info.csv")
    assert_text_file_matches(expected_lines, metadata_filename)

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
