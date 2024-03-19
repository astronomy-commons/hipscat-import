import pandas as pd
from dask.distributed import as_completed
from hipscat import pixel_math
from hipscat.catalog import PartitionInfo
from hipscat.io import file_io, parquet_metadata, paths, write_metadata
from tqdm.auto import tqdm

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
        columns=["partition_order", "partition_pixel", "margin_pixel"],
    )
    return margin_pairs_df


def _create_margin_directory(stats, output_path, storage_options):
    """Creates directories for all the catalog partitions."""
    for healpixel in stats:
        order = healpixel.order
        pix = healpixel.pixel

        destination_dir = paths.pixel_directory(output_path, order, pix)
        file_io.make_directory(destination_dir, exist_ok=True, storage_options=storage_options)


def _map_to_margin_shards(client, args, partition_pixels, margin_pairs):
    """Create all the jobs for mapping partition files into the margin cache."""
    futures = []
    mp_future = client.scatter(margin_pairs, broadcast=True)
    for pix in partition_pixels:
        partition_file = paths.pixel_catalog_file(args.input_catalog_path, pix.order, pix.pixel)
        futures.append(
            client.submit(
                mcmr.map_pixel_shards,
                partition_file=partition_file,
                input_storage_options=args.input_storage_options,
                margin_pairs=mp_future,
                margin_threshold=args.margin_threshold,
                output_path=args.tmp_path,
                margin_order=args.margin_order,
                ra_column=args.catalog.catalog_info.ra_column,
                dec_column=args.catalog.catalog_info.dec_column,
            )
        )

    for _ in tqdm(
        as_completed(futures),
        desc="Mapping  ",
        total=len(futures),
        disable=not args.progress_bar,
    ):
        ...


def _reduce_margin_shards(client, args, partition_pixels):
    """Create all the jobs for reducing margin cache shards into singular files"""
    futures = []

    for pix in partition_pixels:
        futures.append(
            client.submit(
                mcmr.reduce_margin_shards,
                intermediate_directory=args.tmp_path,
                output_path=args.catalog_path,
                output_storage_options=args.output_storage_options,
                partition_order=pix.order,
                partition_pixel=pix.pixel,
                original_catalog_metadata=paths.get_common_metadata_pointer(args.input_catalog_path),
                input_storage_options=args.input_storage_options,
            )
        )

    for _ in tqdm(
        as_completed(futures),
        desc="Reducing ",
        total=len(futures),
        disable=not args.progress_bar,
    ):
        ...


def generate_margin_cache(args, client):
    """Generate a margin cache for a given input catalog.
    The input catalog must be in hipscat format.

    Args:
        args (MarginCacheArguments): A valid `MarginCacheArguments` object.
        client (dask.distributed.Client): A dask distributed client object.
    """
    # determine which order to generate margin pixels for
    partition_stats = args.catalog.partition_info.get_healpix_pixels()

    # get the negative tree pixels
    negative_pixels = args.catalog.generate_negative_tree_pixels()

    combined_pixels = partition_stats + negative_pixels

    margin_pairs = _find_partition_margin_pixel_pairs(combined_pixels, args.margin_order)

    _create_margin_directory(combined_pixels, args.catalog_path, args.output_storage_options)

    _map_to_margin_shards(
        client=client,
        args=args,
        partition_pixels=partition_stats,
        margin_pairs=margin_pairs,
    )

    _reduce_margin_shards(client=client, args=args, partition_pixels=combined_pixels)

    with tqdm(total=4, desc="Finishing", disable=not args.progress_bar) as step_progress:
        parquet_metadata.write_parquet_metadata(
            args.catalog_path, storage_options=args.output_storage_options
        )
        step_progress.update(1)
        total_rows = 0
        metadata_path = paths.get_parquet_metadata_pointer(args.catalog_path)
        for row_group in parquet_metadata.read_row_group_fragments(
            metadata_path, storage_options=args.output_storage_options
        ):
            total_rows += row_group.num_rows
        partition_info = PartitionInfo.read_from_file(
            metadata_path, storage_options=args.output_storage_options
        )
        partition_info_file = paths.get_partition_info_pointer(args.catalog_path)
        partition_info.write_to_file(partition_info_file, storage_options=args.output_storage_options)

        step_progress.update(1)
        margin_catalog_info = args.to_catalog_info(int(total_rows))
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=margin_catalog_info,
            tool_args=args.provenance_info(),
            storage_options=args.output_storage_options,
        )
        write_metadata.write_catalog_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=margin_catalog_info,
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        file_io.remove_directory(
            args.tmp_path, ignore_errors=True, storage_options=args.output_storage_options
        )
        step_progress.update(1)
