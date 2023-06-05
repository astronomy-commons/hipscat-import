import healpy as hp
import numpy as np
import pandas as pd
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
    dec_column
):
    """Creates margin cache shards from a source partition file."""
    data = pd.read_parquet(partition_file)

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
            margin_order=margin_order,
            ra_column=ra_column,
            dec_column=dec_column,
        )

def _to_pixel_shard(data, margin_threshold, output_path, margin_order, ra_column, dec_column):
    """Do boundary checking for the cached partition and then output remaining data."""
    order, pix = data["partition_order"].iloc[0], data["partition_pixel"].iloc[0]
    source_order, source_pix = data["Norder"].iloc[0], data["Npix"].iloc[0]

    scale = pixel_math.get_margin_scale(order, margin_threshold)
    bounding_polygons = pixel_math.get_margin_bounds_and_wcs(order, pix, scale)
    is_polar, _ = pixel_math.pixel_is_polar(order, pix)

    if is_polar:
        data = _margin_filter_polar(
            data,
            order,
            pix,
            margin_order,
            margin_threshold,
            ra_column,
            dec_column,
            bounding_polygons,
        )
    else:
        data["margin_check"] = pixel_math.check_margin_bounds(
            data[ra_column].values,
            data[dec_column].values,
            bounding_polygons
        )

    # pylint: disable-next=singleton-comparison
    margin_data = data.loc[data["margin_check"] == True]

    if len(margin_data):
        # TODO: this should be a utility function in `hipscat`
        # that properly handles the hive formatting
        # generate a file name for our margin shard
        partition_file = paths.pixel_catalog_file(output_path, order, pix)
        partition_dir = f"{partition_file[:-8]}/"
        shard_dir = paths.pixel_directory(
            partition_dir, source_order, source_pix
        )

        file_io.make_directory(shard_dir, exist_ok=True)

        shard_path = paths.pixel_catalog_file(
            partition_dir, source_order, source_pix
        )

        final_df = margin_data.drop(columns=["partition_order", "partition_pixel", "margin_check"])

        if is_polar:
            final_df = final_df.drop(columns=["is_trunc"])

        final_df.to_parquet(shard_path)

def _margin_filter_polar(
    data,
    order,
    pix,
    margin_order,
    margin_threshold,
    ra_column,
    dec_column,
    bounding_polygons
):
    """Filter out margin data around the poles."""
    trunc_pix = pixel_math.get_truncated_margin_pixels(
        order=order,
        pix=pix,
        margin_order=margin_order
    )
    data["is_trunc"] = np.isin(data["margin_pixel"], trunc_pix)

    # pylint: disable=singleton-comparison
    trunc_data = data.loc[data["is_trunc"] == True]
    other_data = data.loc[data["is_trunc"] == False]
    # pylint: enable=singleton-comparison

    trunc_data["margin_check"] = pixel_math.check_polar_margin_bounds(
        trunc_data[ra_column].values,
        trunc_data[dec_column].values,
        order,
        pix,
        margin_order,
        margin_threshold
    )
    other_data["margin_check"] = pixel_math.check_margin_bounds(
        other_data[ra_column].values,
        other_data[dec_column].values,
        bounding_polygons
    )

    return pd.concat([trunc_data, other_data])
