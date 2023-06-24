"""Tests of map reduce operations"""

import os

import hipscat.pixel_math as hist
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest

import hipscat_import.catalog.map_reduce as mr
from hipscat_import.catalog.file_readers import get_file_reader


def test_read_empty_filename():
    """Empty file name"""
    with pytest.raises(FileNotFoundError):
        mr.map_to_pixels(
            input_file="",
            file_reader=get_file_reader("parquet"),
            highest_order=10,
            ra_column="ra",
            dec_column="dec",
        )


def test_read_wrong_fileformat(small_sky_file0):
    """CSV file attempting to be read as parquet"""
    with pytest.raises(pa.lib.ArrowInvalid):
        mr.map_to_pixels(
            input_file=small_sky_file0,
            file_reader=get_file_reader("parquet"),
            highest_order=0,
            ra_column="ra_mean",
            dec_column="dec_mean",
        )


def test_read_directory(test_data_dir):
    """Provide directory, not file"""
    with pytest.raises(FileNotFoundError):
        mr.map_to_pixels(
            input_file=test_data_dir,
            file_reader=get_file_reader("parquet"),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
        )


def test_read_bad_fileformat(blank_data_file):
    """Unsupported file format"""
    with pytest.raises(NotImplementedError):
        mr.map_to_pixels(
            input_file=blank_data_file,
            file_reader=None,
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
        )


def test_read_single_fits(formats_fits):
    """Success case - fits file that exists being read as fits"""
    result = mr.map_to_pixels(
        input_file=formats_fits,
        file_reader=get_file_reader("fits"),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
    )
    assert len(result) == 12
    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)


def test_map_headers_wrong(formats_headers_csv):
    """Test loading the a file with non-default headers (without specifying right headers)"""
    with pytest.raises(ValueError, match="header"):
        mr.map_to_pixels(
            input_file=formats_headers_csv,
            file_reader=get_file_reader("csv"),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
        )


def test_map_headers(formats_headers_csv):
    """Test loading the a file with non-default headers"""
    result = mr.map_to_pixels(
        input_file=formats_headers_csv,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
    )

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 8
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_split_pixels_headers(formats_headers_csv, assert_parquet_file_ids, tmp_path):
    """Test loading the a file with non-default headers"""
    alignment = np.full(12, None)
    alignment[11] = (0, 11, 131)
    mr.split_pixels(
        input_file=formats_headers_csv,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
        shard_suffix=0,
        cache_path=tmp_path,
        alignment=alignment,
    )

    file_name = os.path.join(
        tmp_path, "order_0", "dir_0", "pixel_11", "shard_0_0.parquet"
    )
    expected_ids = [*range(700, 708)]
    assert_parquet_file_ids(file_name, "object_id", expected_ids)

    file_name = os.path.join(
        tmp_path, "order_0", "dir_0", "pixel_1", "shard_0_0.parquet"
    )
    assert not os.path.exists(file_name)


def test_map_small_sky_order0(small_sky_single_file):
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    result = mr.map_to_pixels(
        input_file=small_sky_single_file,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
    )

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_small_sky_part_order1(small_sky_file0):
    """
    Test loading a small portion of the small sky catalog and
    partitioning objects into four smaller buckets
    """
    result = mr.map_to_pixels(
        input_file=small_sky_file0,
        file_reader=get_file_reader("csv"),
        highest_order=1,
        ra_column="ra",
        dec_column="dec",
    )

    assert len(result) == 48

    expected = hist.empty_histogram(1)
    filled_pixels = [5, 7, 11, 2]
    expected[44:] = filled_pixels[:]
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_reduce_order0(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing into one large pixel"""
    mr.reduce_pixel_shards(
        cache_path=parquet_shards_dir,
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_hipscat_index=True,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        delete_input_files=False,
    )

    output_file = os.path.join(tmp_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)


def test_reduce_hipscat_index(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing with or without a _hipscat_index field"""
    mr.reduce_pixel_shards(
        cache_path=parquet_shards_dir,
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        delete_input_files=False,
    )

    output_file = os.path.join(tmp_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)
    # expected_indexes = [
    #     13598131468743213056,
    #     13560933976658411520,
    #     13561582046530240512,
    #     13696722494273093632,
    #     13588709332114997248,
    #     13552942781667737600,
    #     13601023174257934336,
    #     13557123557418336256,
    #     13591216801265483776,
    #     13565852277582856192,
    #     13553697461939208192,
    #     13563711661973438464,
    #     13590818251897569280,
    #     13560168899495854080,
    #     13557816572940124160,
    #     13596001812279721984,
    #     13564690156971098112,
    #     13557377060258709504,
    # ]
    # assert_parquet_file_index(output_file, expected_indexes)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.name == "_hipscat_index"
    npt.assert_array_equal(
        data_frame.columns,
        ["id", "ra", "dec", "ra_error", "dec_error", "Norder", "Dir", "Npix"],
    )

    mr.reduce_pixel_shards(
        cache_path=parquet_shards_dir,
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_hipscat_index=False,  ## different from above
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        delete_input_files=False,
    )

    assert_parquet_file_ids(output_file, "id", expected_ids)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    ## No index name.
    assert data_frame.index.name is None
    ## Data fields are the same.
    npt.assert_array_equal(
        data_frame.columns,
        ["id", "ra", "dec", "ra_error", "dec_error", "Norder", "Dir", "Npix"],
    )


def test_reduce_bad_expectation(parquet_shards_dir, tmp_path):
    """Test reducing into one large pixel"""
    with pytest.raises(ValueError, match="Unexpected number of objects"):
        mr.reduce_pixel_shards(
            cache_path=parquet_shards_dir,
            destination_pixel_order=0,
            destination_pixel_number=11,
            destination_pixel_size=11,  ## should be 131
            output_path=tmp_path,
            ra_column="ra",
            dec_column="dec",
            id_column="id",
            delete_input_files=False,
        )
