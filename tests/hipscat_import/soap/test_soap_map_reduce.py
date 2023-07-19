"""Test components of SOAP"""

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import count_joins, source_to_object_map


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
    map = source_to_object_map(args)

    expected = {
        HealpixPixel(0, 4): [],
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

    assert map == expected


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

    expected = {
        (0, 4): [],
        (2, 176): [(0, 11)],
        (2, 177): [(0, 11)],
        (2, 178): [(0, 11)],
        (2, 179): [(0, 11)],
        (2, 180): [(0, 11)],
        (2, 181): [(0, 11)],
        (2, 182): [(0, 11)],
        (2, 183): [(0, 11)],
        (2, 184): [(0, 11)],
        (2, 185): [(0, 11)],
        (2, 186): [(0, 11)],
        (2, 187): [(0, 11)],
        (1, 47): [(0, 11)],
    }
    for source, objects in expected.items():
        count_joins(args, source, objects)
