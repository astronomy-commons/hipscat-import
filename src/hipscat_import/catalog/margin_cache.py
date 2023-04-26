from hipscat import pixel_math
from hipscat.catalog import Catalog
from hipscat.io import FilePointer, file_io, paths

from dask.distributed import Client, as_completed
import healpy as hp
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

def _find_parition_margin_pixel_pairs(stats, margin_order):
    norders = []
    part_pix = []
    margin_pix = []

    for _, row in stats.iterrows():
        order = row["Norder"]
        pix = row["Npix"]

        d_order = margin_order - order

        margins = pixel_math.get_margin(order, pix, d_order)

        for mp in margins:
            norders.append(order)
            part_pix.append(pix)
            margin_pix.append(mp)

    margin_pairs_df = pd.DataFrame(
        zip(norders, part_pix, margin_pix),
        columns=["partition_order", "partition_pixel", "margin_pixel"]
    )
    return margin_pairs_df

def _create_margin_directory(stats, output_path, margin_name):
    for _, row in stats.iterrows():
        order = row["Norder"]
        pix = row["Npix"]

        destination_dir = paths.pixel_directory(
            output_path, order, pix
        )
        file_io.make_directory(destination_dir, exist_ok=True)

def _map_to_margin_shards(client, partition_files, margin_pairs, margin_threshold, output_path, margin_order, catalog):
    futures = []
    for p in partition_files:
        futures.append(
            client.submit(
                _map_pixel_shards,
                partition_file=p,
                margin_pairs=margin_pairs,
                margin_threshold=margin_threshold,
                output_path=output_path,
                margin_order=margin_order,
                ra_column=catalog.catalog_info.ra_column,
                dec_column=catalog.catalog_info.dec_column,
            )
        )

    for future in tqdm(
        as_completed(futures, with_results=True),
        desc="Mapping  ",
        total=len(futures),
    ):
        ...

def _map_pixel_shards(partition_file, margin_pairs, margin_threshold, output_path, margin_order, ra_column, dec_column):
    data = pd.read_parquet(partition_file)

    data["margin_pixel"] = hp.ang2pix(
        2**margin_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )

    constrained_data = data.merge(margin_pairs, on="margin_pixel")

    constrained_data.groupby(["partition_order", "partition_pixel"]).apply(
        _to_pixel_shard,
        margin_threshold=margin_threshold,
        output_path=output_path,
        margin_order=margin_order,
        ra_column=ra_column,
        dec_column=dec_column,
    )

def _to_pixel_shard(data, margin_threshold, output_path, margin_order, ra_column, dec_column):
    order, pix = data["partition_order"].iloc[0], data["partition_pixel"].iloc[0]
    source_order, source_pix = data["Norder"].iloc[0], data["Npix"].iloc[0]
    pix_dir = int(pix / 10_000) * 10_000
    source_dir = int(source_pix / 10_000) * 10_000

    scale = pixel_math.get_margin_scale(order, margin_threshold)
    bounding_polygons = pixel_math.get_margin_bounds_and_wcs(order, pix, scale)
    is_polar, pole = pixel_math.pixel_is_polar(order, pix)

    if is_polar:
        data = _margin_filter_polar(
            data,
            order,
            pix,
            margin_order,
            pole,
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
        
        margin_data.to_parquet(shard_path)

def _margin_filter_polar(data, order, pix, margin_order, pole, margin_threshold, ra_column, dec_column, bounding_polygons):
    trunc_pix = pixel_math.get_truncated_margin_pixels(order, pix, margin_order, pole)
    data["is_trunc"] = np.isin(data["margin_pixel"], trunc_pix)

    trunc_data = data.loc[data["is_trunc"] == True]
    other_data = data.loc[data["is_trunc"] == False]

    trunc_data["margin_check"] = pixel_math.check_polar_margin_bounds(
        trunc_data[ra_column].values,
        trunc_data[dec_column].values,
        order,
        pix,
        margin_order,
        pole,
        margin_threshold
    )
    other_data["margin_check"] = pixel_math.check_margin_bounds(
        other_data[ra_column].values, 
        other_data[dec_column].values, 
        bounding_polygons
    )

    return pd.concat([trunc_data, other_data])

def generate_margin_cache(catalog_location, margin_threshold, output_path, margin_order=-1):
    with Client(
        n_workers=4,
        threads_per_worker=1,
    ) as client:  # pragma: no cover
        generate_margin_cache_with_client(
            client,
            catalog_location,
            margin_threshold,
            output_path,
            margin_order
        )

def generate_margin_cache_with_client(client, catalog_location, margin_threshold, output_path, margin_order=-1):
    catalog = Catalog(catalog_location)

    # determine which order to generate margin pixels for
    partition_stats = catalog.get_pixels()
    highest_order = np.max(partition_stats["Norder"].values)
    margin_pixel_k = highest_order + 1
    if margin_order > -1:
        if margin_order < margin_pixel_k:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )
        margin_pixel_k = margin_order

    margin_pixel_nside = hp.order2nside(margin_pixel_k)

    if hp.nside2resol(margin_pixel_nside, arcmin=True) / 60. < margin_threshold:
        warnings.warn("Warning: margin pixels have a smaller resolution than margin_threshold; this may lead to data loss in the margin cache.")

    margin_pairs = _find_parition_margin_pixel_pairs(partition_stats, margin_pixel_k)

    margin_name = f"{catalog.catalog_name}_margin_cache_{margin_threshold}"
    base_dir = f"{output_path}{margin_name}"
    _create_margin_directory(partition_stats, base_dir, margin_name)

    partition_files = catalog.partition_info.get_file_names()
    _map_to_margin_shards(client, partition_files, margin_pairs, margin_threshold, base_dir, margin_pixel_k, catalog)


