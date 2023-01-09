"""Convenience class to make it easy to retrieve data files within unit tests"""

import os

TEST_DIR = os.path.dirname(__file__)

TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

# ./blank - one blank file
TEST_BLANK_DATA_DIR = os.path.join(TEST_DATA_DIR, "blank")
TEST_BLANK_CSV = os.path.join(TEST_BLANK_DATA_DIR, "blank.csv")

# ./empty - no files
TEST_EMPTY_DATA_DIR = os.path.join(TEST_DATA_DIR, "empty")

# ./small_sky - all points in one 0-order pixel
TEST_SMALL_SKY_DATA_DIR = os.path.join(TEST_DATA_DIR, "small_sky")
TEST_SMALL_SKY_CSV = os.path.join(TEST_SMALL_SKY_DATA_DIR, "catalog.csv")

# ./small_sky_parts - same as ./small_sky, but broken up into 5 smaller files
TEST_SMALL_SKY_PARTS_DATA_DIR = os.path.join(TEST_DATA_DIR, "small_sky_parts")
TEST_SMALL_SKY_PART0_CSV = os.path.join(
    TEST_SMALL_SKY_PARTS_DATA_DIR, "catalog_00_of_05.csv"
)
TEST_SMALL_SKY_PART1_CSV = os.path.join(
    TEST_SMALL_SKY_PARTS_DATA_DIR, "catalog_01_of_05.csv"
)
TEST_SMALL_SKY_PART2_CSV = os.path.join(
    TEST_SMALL_SKY_PARTS_DATA_DIR, "catalog_02_of_05.csv"
)
TEST_SMALL_SKY_PART3_CSV = os.path.join(
    TEST_SMALL_SKY_PARTS_DATA_DIR, "catalog_03_of_05.csv"
)
TEST_SMALL_SKY_PART4_CSV = os.path.join(
    TEST_SMALL_SKY_PARTS_DATA_DIR, "catalog_04_of_05.csv"
)

# ./parquet_shards - sharded parquet files representing small sky data
TEST_PARQUET_SHARDS_DIR = os.path.join(TEST_DATA_DIR, "parquet_shards")
TEST_PARQUET_SHARDS_DATA_DIR = os.path.join(TEST_PARQUET_SHARDS_DIR, "pixel_44")
TEST_PARQUET_SHARDS_PART0 = os.path.join(
    TEST_PARQUET_SHARDS_DATA_DIR, "shard_0.parquet"
)
TEST_PARQUET_SHARDS_PART1 = os.path.join(
    TEST_PARQUET_SHARDS_DATA_DIR, "shard_1.parquet"
)
TEST_PARQUET_SHARDS_PART2 = os.path.join(
    TEST_PARQUET_SHARDS_DATA_DIR, "shard_2.parquet"
)
TEST_PARQUET_SHARDS_PART3 = os.path.join(
    TEST_PARQUET_SHARDS_DATA_DIR, "shard_3.parquet"
)
TEST_PARQUET_SHARDS_PART4 = os.path.join(
    TEST_PARQUET_SHARDS_DATA_DIR, "shard_4.parquet"
)


# ./test_formats - special formats
TEST_FORMATS_DIR = os.path.join(TEST_DATA_DIR, "test_formats")
TEST_FORMATS_HEADERS_CSV = os.path.join(TEST_FORMATS_DIR, "headers.csv")
TEST_FORMATS_FITS = os.path.join(TEST_FORMATS_DIR, "small_sky.fits")
