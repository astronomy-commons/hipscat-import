import pandas as pd
from dask.distributed import Client, as_completed
from hipscat import pixel_math
from hipscat.io import file_io, paths
from tqdm import tqdm

import hipscat_import.margin_cache.margin_cache_map_reduce as mcmr

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
    mp_future = client.scatter(margin_pairs, broadcast=True)
    for pix in partition_pixels:
        partition_file = paths.pixel_catalog_file(
            args.input_catalog_path,
            pix.order,
            pix.pixel
        )
        futures.append(
            client.submit(
                mcmr.map_pixel_shards,
                partition_file=partition_file,
                margin_pairs=mp_future,
                margin_threshold=args.margin_threshold,
                output_path=args.catalog_path,
                margin_order=args.margin_order,
                ra_column=args.catalog.catalog_info.ra_column,
                dec_column=args.catalog.catalog_info.dec_column,
            )
        )

    for _ in tqdm(
        as_completed(futures),
        desc="Mapping  ",
        total=len(futures),
    ):
        ...

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
