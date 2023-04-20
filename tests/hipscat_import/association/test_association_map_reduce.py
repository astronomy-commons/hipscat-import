"""Test behavior of map reduce methods in association."""

import os

import pandas as pd
import pytest

from hipscat_import.association.map_reduce import reduce_association


def test_reduce_bad_inputs(tmp_path, assert_text_file_matches):
    """Test reducing with corrupted input."""

    input_path = os.path.join(tmp_path, "incomplete_inputs")
    os.makedirs(input_path, exist_ok=True)

    output_path = os.path.join(tmp_path, "output")
    os.makedirs(output_path, exist_ok=True)

    ## We don't even have a partitions file
    with pytest.raises(FileNotFoundError):
        reduce_association(input_path=input_path, output_path=output_path)

    ## Create a partitions file, but it doesn't have the right columns
    partitions_data = pd.DataFrame(
        data=[[700, 282.5, -58.5], [701, 299.5, -48.5]],
        columns=["id", "ra", "dec"],
    )
    partitions_csv_file = os.path.join(input_path, "partitions.csv")
    partitions_data.to_csv(partitions_csv_file, index=False)

    with pytest.raises(KeyError, match="primary_hipscat_index"):
        reduce_association(input_path=input_path, output_path=output_path)

    ## Create a partitions file, but it doesn't have corresponding parquet data.
    partitions_data = pd.DataFrame(
        data=[[0, 0, 11, 0, 0, 11, 131]],
        columns=[
            "Norder",
            "Dir",
            "Npix",
            "join_Norder",
            "join_Dir",
            "join_Npix",
            "primary_hipscat_index",
        ],
    )
    partitions_data.to_csv(partitions_csv_file, index=False)

    with pytest.raises(FileNotFoundError):
        reduce_association(input_path=input_path, output_path=output_path)

    ## We still wrote out the partition info file, though!
    expected_lines = [
        "Norder,Dir,Npix,join_Norder,join_Dir,join_Npix,num_rows",
        "0,0,11,0,0,11,131",
    ]
    metadata_filename = os.path.join(output_path, "partition_join_info.csv")
    assert_text_file_matches(expected_lines, metadata_filename)


def test_reduce_bad_expectation(tmp_path):
    """Test reducing with corrupted input."""
    input_path = os.path.join(tmp_path, "incomplete_inputs")
    os.makedirs(input_path, exist_ok=True)

    output_path = os.path.join(tmp_path, "output")
    os.makedirs(output_path, exist_ok=True)

    ## Create a partitions file, and a parquet file with not-enough rows.
    partitions_data = pd.DataFrame(
        data=[[0, 0, 11, 0, 0, 11, 3]],
        columns=[
            "Norder",
            "Dir",
            "Npix",
            "join_Norder",
            "join_Dir",
            "join_Npix",
            "primary_hipscat_index",
        ],
    )
    partitions_csv_file = os.path.join(input_path, "partitions.csv")
    partitions_data.to_csv(partitions_csv_file, index=False)

    parquet_dir = os.path.join(
        input_path,
        "Norder=0",
        "Dir=0",
        "Npix=11",
        "join_Norder=0",
        "join_Dir=0",
        "join_Npix=11",
    )
    os.makedirs(parquet_dir, exist_ok=True)

    parquet_data = pd.DataFrame(
        data=[[700, 7_000_000, 800, 8_000_000], [701, 7_000_100, 801, 8_001_000]],
        columns=[
            "primary_id",
            "primary_hipscat_index",
            "join_id",
            "join_hipscat_index",
        ],
    )
    parquet_data.to_parquet(os.path.join(parquet_dir, "part0.parquet"))
    with pytest.raises(ValueError, match="Unexpected"):
        reduce_association(input_path=input_path, output_path=output_path)

    ## Add one more row in another file, and the expectation is met.
    parquet_data = pd.DataFrame(
        data=[[702, 7_002_000, 802, 8_002_000]],
        columns=[
            "primary_id",
            "primary_hipscat_index",
            "join_id",
            "join_hipscat_index",
        ],
    )
    parquet_data.to_parquet(os.path.join(parquet_dir, "part1.parquet"))
    reduce_association(input_path=input_path, output_path=output_path)
