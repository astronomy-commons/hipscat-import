import healpy as hp
import pyarrow.dataset as ds
from hipscat import pixel_math
from hipscat.io import file_io, paths

# pylint: disable=too-many-locals,too-many-arguments


def map_pixel_shards(
    partition_file,
    margin_pairs,
    margin_threshold,
    output_path,
    margin_order,
    ra_column,
    dec_column,
):
    """Creates margin cache shards from a source partition file."""
    data = file_io.load_parquet_to_pandas(partition_file)

    data["margin_pixel"] = hp.ang2pix(
        2**margin_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )

    constrained_data = data.merge(margin_pairs, on="margin_pixel")

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
        # TODO: this should be a utility function in `hipscat`
        # that properly handles the hive formatting
        # generate a file name for our margin shard
        partition_dir = _get_partition_directory(output_path, order, pix)
        shard_dir = paths.pixel_directory(partition_dir, source_order, source_pix)

        file_io.make_directory(shard_dir, exist_ok=True)

        shard_path = paths.pixel_catalog_file(partition_dir, source_order, source_pix)

        final_df = margin_data.drop(
            columns=[
                "partition_order",
                "partition_pixel",
                "margin_check",
                "margin_pixel",
            ]
        )

        final_df.to_parquet(shard_path)

        del data, margin_data, final_df


def _get_partition_directory(path, order, pix):
    """Get the directory where a partition pixel's margin shards live"""
    partition_file = paths.pixel_catalog_file(path, order, pix)

    # removes the '.parquet' and adds a slash
    partition_dir = f"{partition_file[:-8]}/"

    return partition_dir


def reduce_margin_shards(output_path, partition_order, partition_pixel):
    """Reduce all partition pixel directories into a single file"""
    shard_dir = _get_partition_directory(output_path, partition_order, partition_pixel)

    if file_io.does_file_or_directory_exist(shard_dir):
        data = ds.dataset(shard_dir, format="parquet")
        full_df = data.to_table().to_pandas()

        if len(full_df):
            margin_cache_file_path = paths.pixel_catalog_file(output_path, partition_order, partition_pixel)

            full_df.to_parquet(margin_cache_file_path)
            file_io.remove_directory(shard_dir)
