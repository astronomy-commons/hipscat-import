"""Test components of SOAP"""

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import (
    combine_partial_results,
    count_joins,
    source_to_object_map,
)


def get_small_sky_maps():
    source_to_object = {
        HealpixPixel(0, 4): [HealpixPixel(0, 11)],
        HealpixPixel(2, 176): [HealpixPixel(0, 11)],
        HealpixPixel(2, 177): [HealpixPixel(0, 11)],
        HealpixPixel(2, 178): [HealpixPixel(0, 11)],
        HealpixPixel(2, 179): [HealpixPixel(0, 11)],
        HealpixPixel(2, 180): [HealpixPixel(0, 11)],
        HealpixPixel(2, 181): [HealpixPixel(0, 11)],
        HealpixPixel(2, 182): [HealpixPixel(0, 11)],
        HealpixPixel(2, 183): [HealpixPixel(0, 11)],
        HealpixPixel(2, 184): [HealpixPixel(0, 11)],
        HealpixPixel(2, 185): [HealpixPixel(0, 11)],
        HealpixPixel(2, 186): [HealpixPixel(0, 11)],
        HealpixPixel(2, 187): [HealpixPixel(0, 11)],
        HealpixPixel(1, 47): [HealpixPixel(0, 11)],
    }

    return source_to_object


def test_source_to_object_map(
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
    source_to_object = source_to_object_map(args)

    expected = get_small_sky_maps()
    assert source_to_object == expected


def test_count_joins(
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

    source_to_object = get_small_sky_maps()
    for source, objects in source_to_object.items():
        count_joins(args, source, objects, tmp_path)

        result = pd.read_csv(
            os.path.join(tmp_path, f"{source.order}_{source.pixel}.csv")
        )
        assert len(result) == 1


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
