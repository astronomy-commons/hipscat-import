import pytest
from hats.pixel_math.healpix_pixel import HealpixPixel

from hats_import.soap.arguments import SoapArguments


@pytest.fixture
def small_sky_soap_maps():
    """Map of source pixels to object+neighbor object pixels"""
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


@pytest.fixture
def small_sky_soap_args(small_sky_object_catalog, small_sky_source_catalog, tmp_path):
    """Default arguments object for association from small sky source to small sky object."""
    return SoapArguments(
        object_catalog_dir=small_sky_object_catalog,
        object_id_column="id",
        source_catalog_dir=small_sky_source_catalog,
        source_object_id_column="object_id",
        output_artifact_name="small_sky_association",
        output_path=tmp_path,
        progress_bar=False,
    )


@pytest.fixture
def catalog_info_data() -> dict:
    return {
        "catalog_name": "test_name",
        "catalog_type": "object",
        "total_rows": 10,
        "ra_column": "ra",
        "dec_column": "dec",
    }


@pytest.fixture
def source_catalog_info() -> dict:
    return {
        "catalog_name": "test_source",
        "catalog_type": "source",
        "total_rows": 100,
        "ra_column": "source_ra",
        "dec_column": "source_dec",
    }
