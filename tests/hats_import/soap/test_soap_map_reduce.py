"""Test components of SOAP"""

import os
import shutil
from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.soap.arguments import SoapArguments
from hats_import.soap.map_reduce import combine_partial_results, count_joins, reduce_joins


def test_count_joins(small_sky_soap_args, tmp_path, small_sky_soap_maps):
    """Test counting association between object and source catalogs."""
    for source, objects in small_sky_soap_maps.items():
        count_joins(small_sky_soap_args, source, objects)

        result = pd.read_csv(
            tmp_path / "small_sky_association" / "intermediate" / f"{source.order}_{source.pixel}.csv"
        )
        assert len(result) == 1
        assert result["num_rows"].sum() > 0


def test_count_joins_with_leaf(small_sky_soap_args, small_sky_soap_maps):
    """Test counting association between object and source catalogs."""
    small_sky_soap_args.write_leaf_files = True
    small_sky_soap_args.source_id_column = "source_id"

    intermediate_dir = Path(small_sky_soap_args.tmp_path)
    for source, objects in small_sky_soap_maps.items():
        count_joins(small_sky_soap_args, source, objects)

        result = pd.read_csv(intermediate_dir / f"{source.order}_{source.pixel}.csv")
        assert len(result) == 1
        assert result["num_rows"].sum() > 0

        parquet_file_name = (
            intermediate_dir
            / "order_0"
            / "dir_0"
            / "pixel_11"
            / f"source_{source.order}_{source.pixel}.parquet"
        )
        assert os.path.exists(parquet_file_name), f"file not found [{parquet_file_name}]"


def test_count_joins_missing(small_sky_source_catalog, tmp_path):
    """Test association between source catalog and itself, where sources are missing from
    either left or right side of the merge."""

    args = SoapArguments(
        object_catalog_dir=small_sky_source_catalog,
        object_id_column="source_id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="source_id",
        output_artifact_name="small_sky_association",
        output_path=tmp_path,
        progress_bar=False,
    )

    ## Pixels don't exist on either side of join.
    source = HealpixPixel(0, 6)
    with pytest.raises(FileNotFoundError):
        count_joins(args, source, [])

    ## Pixels are totally non-overlapping
    source = HealpixPixel(2, 176)
    count_joins(args, source, [HealpixPixel(2, 177), HealpixPixel(2, 178)])

    result_csv = tmp_path / "small_sky_association" / "intermediate" / f"{source.order}_{source.pixel}.csv"

    result = pd.read_csv(result_csv)
    assert len(result) == 3
    assert result["num_rows"][0:1].sum() == 0
    expected = [[2, 177, 0], [2, 178, 0], [-1, -1, 385]]
    npt.assert_array_equal(result[["Norder", "Npix", "num_rows"]].to_numpy(), expected)

    ## We send more pixels to match than we need - not all are returned in results
    source = HealpixPixel(2, 176)
    count_joins(args, source, [HealpixPixel(2, 176), HealpixPixel(2, 177), HealpixPixel(2, 178)])

    result = pd.read_csv(result_csv)
    assert len(result) == 1
    expected = [[2, 176, 385]]
    npt.assert_array_equal(result[["Norder", "Npix", "num_rows"]].to_numpy(), expected)


def test_combine_results(tmp_path):
    """Test combining many CSVs into a single one"""
    input_path = tmp_path / "input"
    input_path.mkdir(parents=True)

    output_path = tmp_path / "output"
    output_path.mkdir(parents=True)

    join_info = pd.DataFrame(
        data=[
            [0, 0, 11, 0, 0, 11, 130],
            [0, 0, 11, 0, 0, 0, 1],
            [-1, 0, -1, 0, 0, 0, 1],
        ],
        columns=[
            "Norder",
            "Dir",
            "Npix",
            "join_Norder",
            "join_Dir",
            "join_Npix",
            "num_rows",
        ],
    )
    partitions_csv_file = input_path / "0_11.csv"
    join_info.to_csv(partitions_csv_file, index=False)

    total_num_rows = combine_partial_results(input_path, output_path)
    assert total_num_rows == 131

    result = pd.read_csv(output_path / "partition_join_info.csv")
    assert len(result) == 2

    result = pd.read_csv(output_path / "unmatched_sources.csv")
    assert len(result) == 1


def test_reduce_joins(small_sky_soap_args, soap_intermediate_dir, small_sky_soap_maps):
    """Use some previously-computed intermediate files to reduce the joined
    leaf parquet files into a single parquet file."""
    temp_path = os.path.join(small_sky_soap_args.tmp_path, "resume", "intermediate")
    shutil.copytree(
        soap_intermediate_dir,
        temp_path,
    )
    os.makedirs(os.path.join(temp_path, "reducing"))
    small_sky_soap_args.tmp_path = temp_path

    reduce_joins(small_sky_soap_args, HealpixPixel(0, 11), object_key="0_11")

    parquet_file_name = (
        small_sky_soap_args.catalog_path / "dataset" / "Norder=0" / "Dir=0" / "Npix=11.parquet"
    )
    assert os.path.exists(parquet_file_name), f"file not found [{parquet_file_name}]"

    parquet_file = pq.ParquetFile(parquet_file_name)
    assert parquet_file.metadata.num_row_groups == 14
    assert parquet_file.metadata.num_rows == 17161
    assert parquet_file.metadata.num_columns == 9
    join_order_column = parquet_file.metadata.num_columns - 4
    join_pix_column = parquet_file.metadata.num_columns - 2

    ## Check that the row groups inside the parquet file are ordered "breadth-first".
    ## Fetch the list of healpix pixels using the parquet metadata.
    ordered_pixels = [
        HealpixPixel(
            parquet_file.metadata.row_group(row_index).column(join_order_column).statistics.min,
            parquet_file.metadata.row_group(row_index).column(join_pix_column).statistics.min,
        )
        for row_index in range(14)
    ]
    assert ordered_pixels == list(small_sky_soap_maps.keys())


def test_reduce_joins_missing_files(small_sky_soap_args, soap_intermediate_dir, capsys):
    """Use some previously-computed intermediate files to reduce the joined
    leaf parquet files into a single parquet file."""
    temp_path = os.path.join(small_sky_soap_args.tmp_path, "resume", "intermediate")
    shutil.copytree(
        soap_intermediate_dir,
        temp_path,
    )
    small_sky_soap_args.tmp_path = temp_path

    with pytest.raises(FileNotFoundError):
        reduce_joins(small_sky_soap_args, HealpixPixel(0, 11), object_key="0_11")
    captured = capsys.readouterr()
    assert "No such file or directory" in captured.out
