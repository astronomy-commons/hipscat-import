import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from hipscat import pixel_math
from hipscat.catalog.partition_info import PartitionInfo
from hipscat.io import file_io, paths
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN

from hipscat_import.pipeline_resume_plan import get_pixel_cache_directory


def map_pixel_shards(
    partition_file,
    input_storage_options,
    margin_pairs,
    margin_threshold,
    output_path,
    margin_order,
    ra_column,
    dec_column,
):
    """Creates margin cache shards from a source partition file."""
    data = file_io.load_parquet_to_pandas(partition_file, storage_options=input_storage_options)

    data["margin_pixel"] = hp.ang2pix(
        2**margin_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )

    constrained_data = data.reset_index().merge(margin_pairs, on="margin_pixel")

    if len(constrained_data):
        constrained_data.groupby(["partition_order", "partition_pixel"]).apply(
            _to_pixel_shard,
            margin_threshold=margin_threshold,
            output_path=output_path,
            ra_column=ra_column,
            dec_column=dec_column,
        )


def _to_pixel_shard(data, margin_threshold, output_path, ra_column, dec_column):
    """Do boundary checking for the cached partition and then output remaining data."""
    order, pix = data["partition_order"].iloc[0], data["partition_pixel"].iloc[0]
    source_order, source_pix = data["Norder"].iloc[0], data["Npix"].iloc[0]

    data["margin_check"] = pixel_math.check_margin_bounds(
        data[ra_column].values, data[dec_column].values, order, pix, margin_threshold
    )

    # pylint: disable-next=singleton-comparison
    margin_data = data.loc[data["margin_check"] == True]

    if len(margin_data):
        # generate a file name for our margin shard, that uses both sets of Norder/Npix
        partition_dir = get_pixel_cache_directory(output_path, HealpixPixel(order, pix))
        shard_dir = paths.pixel_directory(partition_dir, source_order, source_pix)

        file_io.make_directory(shard_dir, exist_ok=True)

        shard_path = paths.pixel_catalog_file(partition_dir, source_order, source_pix)

        final_df = margin_data.drop(
            columns=[
                "margin_check",
                "margin_pixel",
            ]
        )

        rename_columns = {
            PartitionInfo.METADATA_ORDER_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_ORDER_COLUMN_NAME}",
            PartitionInfo.METADATA_DIR_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_DIR_COLUMN_NAME}",
            PartitionInfo.METADATA_PIXEL_COLUMN_NAME: f"margin_{PartitionInfo.METADATA_PIXEL_COLUMN_NAME}",
            "partition_order": PartitionInfo.METADATA_ORDER_COLUMN_NAME,
            "partition_pixel": PartitionInfo.METADATA_PIXEL_COLUMN_NAME,
        }

        final_df.rename(columns=rename_columns, inplace=True)

        dir_column = np.floor_divide(final_df[PartitionInfo.METADATA_PIXEL_COLUMN_NAME].values, 10000) * 10000

        final_df[PartitionInfo.METADATA_DIR_COLUMN_NAME] = dir_column

        final_df = final_df.astype(
            {
                PartitionInfo.METADATA_ORDER_COLUMN_NAME: np.uint8,
                PartitionInfo.METADATA_DIR_COLUMN_NAME: np.uint64,
                PartitionInfo.METADATA_PIXEL_COLUMN_NAME: np.uint64,
            }
        )
        final_df = final_df.set_index(HIPSCAT_ID_COLUMN).sort_index()

        final_df.to_parquet(shard_path)

        del data, margin_data, final_df


def reduce_margin_shards(
    intermediate_directory,
    output_path,
    output_storage_options,
    partition_order,
    partition_pixel,
    original_catalog_metadata,
    input_storage_options,
):
    """Reduce all partition pixel directories into a single file"""
    shard_dir = get_pixel_cache_directory(
        intermediate_directory, HealpixPixel(partition_order, partition_pixel)
    )
    if file_io.does_file_or_directory_exist(shard_dir):
        data = ds.dataset(shard_dir, format="parquet")
        full_df = data.to_table().to_pandas()
        margin_cache_dir = paths.pixel_directory(output_path, partition_order, partition_pixel)
        file_io.make_directory(margin_cache_dir, exist_ok=True, storage_options=output_storage_options)

        if len(full_df):
            schema = file_io.read_parquet_metadata(
                original_catalog_metadata, storage_options=input_storage_options
            ).schema.to_arrow_schema()

            schema = (
                schema.append(pa.field("margin_Norder", pa.uint8()))
                .append(pa.field("margin_Dir", pa.uint64()))
                .append(pa.field("margin_Npix", pa.uint64()))
            )

            margin_cache_file_path = paths.pixel_catalog_file(output_path, partition_order, partition_pixel)

            full_df.to_parquet(margin_cache_file_path, schema=schema, storage_options=output_storage_options)
            file_io.remove_directory(shard_dir)
