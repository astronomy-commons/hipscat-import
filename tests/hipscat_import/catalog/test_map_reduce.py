"""Tests of map reduce operations"""

import os
from io import StringIO

import healpy as hp
import hipscat.pixel_math as hist
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest
from numpy import frombuffer

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
            resume_path="",
            mapping_key="map_0",
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
            resume_path="",
            mapping_key="map_0",
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
            resume_path="",
            mapping_key="map_0",
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
            resume_path="",
            mapping_key="map_0",
        )


def read_partial_histogram(tmp_path, mapping_key):
    """Helper to read in the former result of a map operation."""
    histogram_file = os.path.join(tmp_path, "histograms", f"{mapping_key}.binary")
    with open(histogram_file, "rb") as file_handle:
        return frombuffer(file_handle.read(), dtype=np.int64)


def test_read_single_fits(tmp_path, formats_fits):
    """Success case - fits file that exists being read as fits"""
    os.makedirs(os.path.join(tmp_path, "histograms"))
    mr.map_to_pixels(
        input_file=formats_fits,
        file_reader=get_file_reader("fits"),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")
    assert len(result) == 12
    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)


def test_map_headers_wrong(formats_headers_csv):
    """Test loading the a file with non-default headers (without specifying right headers)"""
    with pytest.raises(ValueError, match="columns expected but not found"):
        mr.map_to_pixels(
            input_file=formats_headers_csv,
            file_reader=get_file_reader("csv"),
            highest_order=0,
            ra_column="ra",
            dec_column="dec",
            resume_path="",
            mapping_key="map_0",
        )


def test_map_headers(tmp_path, formats_headers_csv):
    """Test loading the a file with non-default headers"""
    os.makedirs(os.path.join(tmp_path, "histograms"))
    mr.map_to_pixels(
        input_file=formats_headers_csv,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 8
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_with_hipscat_index(tmp_path, formats_dir, small_sky_single_file):
    os.makedirs(os.path.join(tmp_path, "histograms"))
    input_file = os.path.join(formats_dir, "hipscat_index.csv")
    mr.map_to_pixels(
        input_file=input_file,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="NOPE",
        dec_column="NOPE",
        use_hipscat_index=True,  # radec don't matter. just use existing index
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    expected = hist.empty_histogram(0)
    expected[11] = 131

    result = read_partial_histogram(tmp_path, "map_0")
    npt.assert_array_equal(result, expected)

    with pytest.raises(ValueError, match="columns expected but not found"):
        mr.map_to_pixels(
            input_file=small_sky_single_file,
            file_reader=get_file_reader("csv"),
            highest_order=0,
            ra_column="NOPE",
            dec_column="NOPE",
            use_hipscat_index=True,  # no pre-existing index! expect failure.
            resume_path=tmp_path,
            mapping_key="map_0",
        )


def test_map_with_schema(tmp_path, mixed_schema_csv_dir, mixed_schema_csv_parquet):
    """Test loading the a file when using a parquet schema file for dtypes"""
    os.makedirs(os.path.join(tmp_path, "histograms"))
    input_file = os.path.join(mixed_schema_csv_dir, "input_01.csv")
    mr.map_to_pixels(
        input_file=input_file,
        file_reader=get_file_reader(
            "csv",
            schema_file=mixed_schema_csv_parquet,
        ),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 4
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_small_sky_order0(tmp_path, small_sky_single_file):
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    os.makedirs(os.path.join(tmp_path, "histograms"))
    mr.map_to_pixels(
        input_file=small_sky_single_file,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 12

    expected = hist.empty_histogram(0)
    expected[11] = 131
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_map_small_sky_part_order1(tmp_path, small_sky_file0):
    """
    Test loading a small portion of the small sky catalog and
    partitioning objects into four smaller buckets
    """
    os.makedirs(os.path.join(tmp_path, "histograms"))
    mr.map_to_pixels(
        input_file=small_sky_file0,
        file_reader=get_file_reader("csv"),
        highest_order=1,
        ra_column="ra",
        dec_column="dec",
        resume_path=tmp_path,
        mapping_key="map_0",
    )

    result = read_partial_histogram(tmp_path, "map_0")

    assert len(result) == 48

    expected = hist.empty_histogram(1)
    filled_pixels = [5, 7, 11, 2]
    expected[44:] = filled_pixels[:]
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_split_pixels_headers(formats_headers_csv, assert_parquet_file_ids, tmp_path):
    """Test loading the a file with non-default headers"""
    os.makedirs(os.path.join(tmp_path, "splitting"))
    alignment = np.full(12, None)
    alignment[11] = (0, 11, 131)
    mr.split_pixels(
        input_file=formats_headers_csv,
        file_reader=get_file_reader("csv"),
        highest_order=0,
        ra_column="ra_mean",
        dec_column="dec_mean",
        splitting_key="0",
        cache_shard_path=tmp_path,
        resume_path=tmp_path,
        alignment=alignment,
    )

    file_name = os.path.join(tmp_path, "order_0", "dir_0", "pixel_11", "shard_0_0.parquet")
    expected_ids = [*range(700, 708)]
    assert_parquet_file_ids(file_name, "object_id", expected_ids)

    file_name = os.path.join(tmp_path, "order_0", "dir_0", "pixel_1", "shard_0_0.parquet")
    assert not os.path.exists(file_name)


def test_reduce_order0(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing into one large pixel"""
    os.makedirs(os.path.join(tmp_path, "reducing"))
    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_hipscat_index=True,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    output_file = os.path.join(tmp_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)


def test_reduce_hipscat_index(parquet_shards_dir, assert_parquet_file_ids, tmp_path):
    """Test reducing with or without a _hipscat_index field"""
    os.makedirs(os.path.join(tmp_path, "reducing"))
    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
        delete_input_files=False,
    )

    output_file = os.path.join(tmp_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    expected_ids = [*range(700, 831)]
    assert_parquet_file_ids(output_file, "id", expected_ids)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    assert data_frame.index.name == "_hipscat_index"
    npt.assert_array_equal(
        data_frame.columns,
        ["id", "ra", "dec", "ra_error", "dec_error", "Norder", "Dir", "Npix"],
    )

    mr.reduce_pixel_shards(
        cache_shard_path=parquet_shards_dir,
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=131,
        output_path=tmp_path,
        add_hipscat_index=False,  ## different from above
        ra_column="ra",
        dec_column="dec",
        sort_columns="id",
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
            cache_shard_path=parquet_shards_dir,
            resume_path=tmp_path,
            reducing_key="0_11",
            destination_pixel_order=0,
            destination_pixel_number=11,
            destination_pixel_size=11,  ## should be 131
            output_path=tmp_path,
            ra_column="ra",
            dec_column="dec",
            sort_columns="id",
            delete_input_files=False,
        )


def test_reduce_with_sorting_complex(assert_parquet_file_ids, tmp_path):
    """Test reducing and requesting specific sort columns.

    Logically, the input data has a mix of orderings in files, object IDs, and timestamps.
    Each source is partitioned according to the linked object's radec, and so will be
    ordered within the same hipscat_index value.

    First, we take some time to set up these silly data points, then we test out
    reducing them into a single parquet file using a mix of reduction options.
    """
    os.makedirs(os.path.join(tmp_path, "reducing"))
    shard_dir = os.path.join(tmp_path, "reduce_shards", "order_0", "dir_0", "pixel_11")
    os.makedirs(shard_dir)
    output_file = os.path.join(tmp_path, "Norder=0", "Dir=0", "Npix=11.parquet")

    file1_string = """source_id,object_id,time,ra,dec
1200,700,3000,282.5,-58.5
1201,700,4000,282.5,-58.5
1402,702,3000,310.5,-27.5
1403,702,3100,310.5,-27.5
1404,702,3200,310.5,-27.5
1505,703,4000,286.5,-69.5"""
    file1_data = pd.read_csv(StringIO(file1_string))
    file1_data.to_parquet(os.path.join(shard_dir, "file_1_shard_1.parquet"))

    file2_string = """source_id,object_id,time,ra,dec
1206,700,2000,282.5,-58.5
1307,701,2200,299.5,-48.5
1308,701,2100,299.5,-48.5
1309,701,2000,299.5,-48.5"""
    file2_data = pd.read_csv(StringIO(file2_string))
    file2_data.to_parquet(os.path.join(shard_dir, "file_2_shard_1.parquet"))

    combined_data = pd.concat([file1_data, file2_data])
    combined_data["norder19_healpix"] = hp.ang2pix(
        2**19,
        combined_data["ra"].values,
        combined_data["dec"].values,
        lonlat=True,
        nest=True,
    )
    ## Use this to prune generated columns like Norder, Npix, and _hipscat_index
    comparison_columns = ["source_id", "object_id", "time", "ra", "dec"]

    ######################## Sort option 1: by source_id
    ## This will sort WITHIN an order 19 healpix pixel. In that ordering, the objects are
    ## (703, 700, 701, 702)
    mr.reduce_pixel_shards(
        cache_shard_path=os.path.join(tmp_path, "reduce_shards"),
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="source_id",
        delete_input_files=False,
    )

    ## sort order is effectively (norder19 healpix, source_id)
    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["norder19_healpix", "source_id"])
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1505, 1200, 1201, 1206, 1307, 1308, 1309, 1402, 1403, 1404],
        resort_ids=False,
    )

    assert_parquet_file_ids(
        output_file,
        "object_id",
        [703, 700, 700, 700, 701, 701, 701, 702, 702, 702],
        resort_ids=False,
    )

    ######################## Sort option 2: by object id and time
    ## sort order is effectively (norder19 healpix, object id, time)
    mr.reduce_pixel_shards(
        cache_shard_path=os.path.join(tmp_path, "reduce_shards"),
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="object_id,time",
        delete_input_files=False,
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["norder19_healpix", "object_id", "time"])
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1505, 1206, 1200, 1201, 1309, 1308, 1307, 1402, 1403, 1404],
        resort_ids=False,
    )
    assert_parquet_file_ids(
        output_file,
        "time",
        [4000, 2000, 3000, 4000, 2000, 2100, 2200, 3000, 3100, 3200],
        resort_ids=False,
    )

    ######################## Sort option 3: by object id and time WITHOUT hipscat index.
    ## The 1500 block of ids goes back to the end, because we're not using
    ## spatial properties for sorting, only numeric.
    ## sort order is effectively (object id, time)
    mr.reduce_pixel_shards(
        cache_shard_path=os.path.join(tmp_path, "reduce_shards"),
        resume_path=tmp_path,
        reducing_key="0_11",
        destination_pixel_order=0,
        destination_pixel_number=11,
        destination_pixel_size=10,
        output_path=tmp_path,
        ra_column="ra",
        dec_column="dec",
        sort_columns="object_id,time",
        add_hipscat_index=False,
        delete_input_files=False,
    )

    data_frame = pd.read_parquet(output_file, engine="pyarrow")
    expected_dataframe = combined_data.sort_values(["object_id", "time"])
    pd.testing.assert_frame_equal(
        expected_dataframe[comparison_columns].reset_index(drop=True),
        data_frame[comparison_columns].reset_index(drop=True),
    )
    assert_parquet_file_ids(
        output_file,
        "source_id",
        [1206, 1200, 1201, 1309, 1308, 1307, 1402, 1403, 1404, 1505],
        resort_ids=False,
    )
