"""Inner methods for SOAP"""

import healpy as hp
import numpy as np
import pandas as pd
from hipscat.catalog import Catalog
from hipscat.io import file_io
from hipscat.io.paths import pixel_catalog_file
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_tree import PixelAlignment, align_trees


def source_to_object_map(args):
    """Build a map of (source order/pixel) to the (object order/pixel)
    that are aligned.
    """
    object_catalog = Catalog.read_from_hipscat(args.object_catalog_dir)
    source_catalog = Catalog.read_from_hipscat(args.source_catalog_dir)

    ## Direct aligment from source to object
    ###############################################
    alignment = align_trees(
        object_catalog.pixel_tree, source_catalog.pixel_tree, "outer"
    )
    som = alignment.pixel_mapping
    som = som.groupby(
        [
            PixelAlignment.JOIN_ORDER_COLUMN_NAME,
            PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
        ],
        group_keys=False,
    )

    ## Lots of cute comprehension is happening here.
    ## create tuple of (source order/pixel) and [array of tuples of (object order/pixel)]
    source_to_object = [
        (
            HealpixPixel(int(source_name[0]), int(source_name[1])),
            [
                HealpixPixel(int(object_elem[0]), int(object_elem[1]))
                for object_elem in object_group.dropna().to_numpy().T[:2].T
            ],
        )
        for source_name, object_group in som
    ]
    ## Treat the array of tuples as a dictionary.
    source_to_object = dict(source_to_object)

    ## Object neighbors for source
    ###############################################
    max_order = max(
        object_catalog.partition_info.get_highest_order(),
        source_catalog.partition_info.get_highest_order(),
    )

    object_order_map = np.full(hp.order2npix(max_order), -1)

    for pixel in object_catalog.partition_info.get_healpix_pixels():
        explosion_factor = 4 ** (max_order - pixel.order)
        exploded_pixels = [
            *range(
                pixel.pixel * explosion_factor,
                (pixel.pixel + 1) * explosion_factor,
            )
        ]
        object_order_map[exploded_pixels] = pixel.order

    source_to_neighbor_object = {}

    for source, objects in source_to_object.items():
        # get all neighboring pixels
        nside = hp.order2nside(source.order)
        neighbors = hp.get_all_neighbours(nside, source.pixel, nest=True)

        ## get rid of -1s and normalize to max order
        explosion_factor = 4 ** (max_order - source.order)
        neighbors = [
            neighbor * explosion_factor for neighbor in neighbors if neighbor != -1
        ]

        neighbors_orders = object_order_map[neighbors]
        desploded = [
            HealpixPixel(order, hoo_pixel >> 2 * (max_order - order))
            for order, hoo_pixel in list(zip(neighbors_orders, neighbors))
            if order != -1
        ]
        neighbors = set(desploded) - set(objects)
        source_to_neighbor_object[source] = objects + list(neighbors)

    return source_to_neighbor_object


def _count_joins_for_object(
    source_data, object_catalog_dir, object_id_column, object_pixel
):
    object_path = pixel_catalog_file(
        catalog_base_dir=object_catalog_dir,
        pixel_order=object_pixel.order,
        pixel_number=object_pixel.pixel,
    )
    object_data = pd.read_parquet(
        path=object_path, columns=[object_id_column]
    ).set_index(object_id_column)

    joined_data = source_data.merge(
        object_data, how="inner", left_index=True, right_index=True
    )

    return len(joined_data)


def _write_count_results(cache_path, source_healpix, results):
    """Build a nice dataframe with pretty columns and rows"""
    num_results = len(results)
    dataframe = pd.DataFrame(results, columns=["Norder", "Npix", "num_rows"])

    dataframe["Dir"] = [
        int(order / 10_000) * 10_000 if order >= 0 else -1 for order, _, _ in results
    ]
    dataframe["join_Norder"] = np.full(
        num_results, fill_value=source_healpix.order, dtype=np.int32
    )
    dataframe["join_Dir"] = [
        int(order / 10_000) * 10_000 for order in dataframe["join_Norder"]
    ]
    dataframe["join_Npix"] = np.full(
        num_results, fill_value=source_healpix.pixel, dtype=np.int32
    )

    ## Reorder columns.
    dataframe = dataframe[
        ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]
    ]

    file_io.write_dataframe_to_csv(
        dataframe=dataframe,
        file_pointer=file_io.append_paths_to_pointer(
            cache_path, f"{source_healpix.order}_{source_healpix.pixel}.csv"
        ),
        index=False,
    )


def count_joins(soap_args, source_pixel, object_pixels, cache_path):
    """Count the number of equijoined sources in the object pixels.

    If any un-joined source pixels remain, stretch out to neighboring object pixels.
    """
    source_path = pixel_catalog_file(
        catalog_base_dir=soap_args.source_catalog_dir,
        pixel_order=source_pixel.order,
        pixel_number=source_pixel.pixel,
    )
    source_data = pd.read_parquet(
        path=source_path, columns=[soap_args.source_object_id_column]
    ).set_index(soap_args.source_object_id_column)

    remaining_sources = len(source_data)
    results = []

    for object_pixel in object_pixels:
        if remaining_sources < 1:
            break
        join_count = _count_joins_for_object(
            source_data,
            soap_args.object_catalog_dir,
            soap_args.object_id_column,
            object_pixel,
        )
        results.append([object_pixel.order, object_pixel.pixel, join_count])
        remaining_sources -= join_count

    ## mark that some sources were not joined
    if remaining_sources > 0:
        results.append([-1, -1, remaining_sources])

    _write_count_results(cache_path, source_pixel, results)


def combine_partial_results(input_path, output_path):
    """Combine many partial CSVs into single partition join info.

    Also write out a debug file with counts of unmatched sources, if any.
    """
    partial_files = file_io.find_files_matching_path(input_path, "**.csv")
    partials = []

    for partial_file in partial_files:
        partials.append(file_io.load_csv_to_pandas(partial_file))

    dataframe = pd.concat(partials)

    matched = dataframe.loc[dataframe["Norder"] != -1]
    unmatched = dataframe.loc[dataframe["Norder"] == -1]

    file_io.write_dataframe_to_csv(
        dataframe=matched,
        file_pointer=file_io.append_paths_to_pointer(
            output_path, "partition_join_info.csv"
        ),
        index=False,
    )

    if len(unmatched) > 0:
        file_io.write_dataframe_to_csv(
            dataframe=unmatched,
            file_pointer=file_io.append_paths_to_pointer(
                output_path, "unmatched_sources.csv"
            ),
            index=False,
        )
