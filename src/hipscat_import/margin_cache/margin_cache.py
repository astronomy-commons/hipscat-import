import healpy as hp
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from hipscat import pixel_math
from hipscat.io import file_io, paths
from tqdm import tqdm

# pylint: disable=too-many-locals,too-many-arguments

def _find_partition_margin_pixel_pairs(stats, margin_order):
    """Creates a DataFrame filled with many-to-many connections between
    the catalog partition pixels and the margin pixels at `margin_order`.
    """
    norders = []
    part_pix = []
    margin_pix = []

    for healpixel in stats:
        order = healpixel.order
        pix = healpixel.pixel

        d_order = margin_order - order

        margins = pixel_math.get_margin(order, pix, d_order)

        for m_p in margins:
            norders.append(order)
            part_pix.append(pix)
            margin_pix.append(m_p)

    margin_pairs_df = pd.DataFrame(
        zip(norders, part_pix, margin_pix),
        columns=["partition_order", "partition_pixel", "margin_pixel"]
    )
    return margin_pairs_df

def _create_margin_directory(stats, output_path):
    """Creates directories for all the catalog partitions."""
    for healpixel in stats:
        order = healpixel.order
        pix = healpixel.pixel

        destination_dir = paths.pixel_directory(
            output_path, order, pix
        )
        file_io.make_directory(destination_dir, exist_ok=True)

def _map_to_margin_shards(client, args, partition_pixels, margin_pairs):
    """Create all the jobs for mapping partition files into the margin cache."""
    futures = []
    for pix in partition_pixels:
        partition_file = paths.pixel_catalog_file(
            args.input_catalog_path,
            pix.order,
            pix.pixel
        )
        futures.append(
            client.submit(
                _map_pixel_shards,
                partition_file=partition_file,
                margin_pairs=margin_pairs,
                margin_threshold=args.margin_threshold,
                output_path=args.catalog_path,
                margin_order=args.margin_order,
                ra_column=args.catalog.catalog_info.ra_column,
                dec_column=args.catalog.catalog_info.dec_column,
            )
        )

    for _ in tqdm(
        as_completed(futures, with_results=True),
        desc="Mapping  ",
        total=len(futures),
    ):
        ...

def _map_pixel_shards(
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

def generate_margin_cache(args):
    """Generate a margin cache for a given input catalog.
    The input catalog must be in hipscat format.
    This method will handle the creation of the `dask.distributed` client
    based on the `dask_tmp`, `dask_n_workers`, and `dask_threads_per_worker`
    values of the `MarginCacheArguments` object.

    Args:
        args (MarginCacheArguments): A valid `MarginCacheArguments` object.
    """
    # pylint: disable=duplicate-code
    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:  # pragma: no cover
        generate_margin_cache_with_client(
            client,
            args
        )
    # pylint: enable=duplicate-code

def generate_margin_cache_with_client(client, args):
    """Generate a margin cache for a given input catalog.
    The input catalog must be in hipscat format.
    Args:
        client (dask.distributed.Client): A dask distributed client object.
        args (MarginCacheArguments): A valid `MarginCacheArguments` object.
    """
    # determine which order to generate margin pixels for
    partition_stats = args.catalog.partition_info.get_healpix_pixels()

    margin_pairs = _find_partition_margin_pixel_pairs(
        partition_stats,
        args.margin_order
    )

    # arcsec to degree conversion
    # TODO: remove this once hipscat uses arcsec for calculation
    args.margin_threshold = args.margin_threshold / 3600.

    _create_margin_directory(
        partition_stats, args.catalog_path
    )

    _map_to_margin_shards(
        client=client,
        args=args,
        partition_pixels=partition_stats,
        margin_pairs=margin_pairs,
    )
