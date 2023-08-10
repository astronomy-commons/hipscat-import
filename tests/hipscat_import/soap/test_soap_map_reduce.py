"""Test components of SOAP"""

import os

import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import combine_partial_results, count_joins


def test_count_joins(small_sky_soap_args, tmp_path, small_sky_soap_maps):
    """Test counting association between object and source catalogs."""
    for source, objects in small_sky_soap_maps.items():
        count_joins(small_sky_soap_args, source, objects, tmp_path)

        result = pd.read_csv(os.path.join(tmp_path, f"{source.order}_{source.pixel}.csv"))
        assert len(result) == 1
        assert result["num_rows"].sum() > 0


def test_count_joins_missing(small_sky_source_catalog, tmp_path):
    """Test association between source catalog and itself, where sources are missing from
    either left or right side of the merge."""

    args = SoapArguments(
        object_catalog_dir=small_sky_source_catalog,
        object_id_column="source_id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="source_id",
        output_catalog_name="small_sky_association",
        output_path=tmp_path,
        progress_bar=False,
    )

    ## Pixels don't exist on either side of join.
    source = HealpixPixel(0, 6)
    with pytest.raises(FileNotFoundError):
        count_joins(args, source, [], tmp_path)

    ## Pixels are totally non-overlapping
    source = HealpixPixel(2, 176)
    count_joins(args, source, [HealpixPixel(2, 177), HealpixPixel(2, 178)], tmp_path)

    result_csv = os.path.join(tmp_path, f"{source.order}_{source.pixel}.csv")

    result = pd.read_csv(result_csv)
    assert len(result) == 3
    assert result["num_rows"][0:1].sum() == 0
    expected = [[2, 177, 0], [2, 178, 0], [-1, -1, 385]]
    npt.assert_array_equal(result[["Norder", "Npix", "num_rows"]].to_numpy(), expected)

    ## We send more pixels to match than we need - not all are returned in results
    source = HealpixPixel(2, 176)
    count_joins(args, source, [HealpixPixel(2, 176), HealpixPixel(2, 177), HealpixPixel(2, 178)], tmp_path)

    result = pd.read_csv(result_csv)
    assert len(result) == 1
    expected = [[2, 176, 385]]
    npt.assert_array_equal(result[["Norder", "Npix", "num_rows"]].to_numpy(), expected)


def test_combine_results(tmp_path):
    """Test combining many CSVs into a single one"""
    input_path = os.path.join(tmp_path, "input")
    os.makedirs(input_path, exist_ok=True)

    output_path = os.path.join(tmp_path, "output")
    os.makedirs(output_path, exist_ok=True)

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
    partitions_csv_file = os.path.join(input_path, "0_11.csv")
    join_info.to_csv(partitions_csv_file, index=False)

    combine_partial_results(input_path, output_path)

    result = pd.read_csv(os.path.join(output_path, "partition_join_info.csv"))
    assert len(result) == 2

    result = pd.read_csv(os.path.join(output_path, "unmatched_sources.csv"))
    assert len(result) == 1
