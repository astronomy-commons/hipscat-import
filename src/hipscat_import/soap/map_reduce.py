"""Inner methods for SOAP"""

import pandas as pd
from hipscat.catalog import Catalog
from hipscat.io.paths import pixel_catalog_file
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_tree.pixel_alignment import PixelAlignment, align_trees


def source_to_object_map(args):
    """Build a map of (source order/pixel) to the (object order/pixel)
    that are aligned.
    """
    object_catalog = Catalog.read_from_hipscat(args.object_catalog_dir)
    source_catalog = Catalog.read_from_hipscat(args.source_catalog_dir)
    alignment = align_trees(
        object_catalog.pixel_tree, source_catalog.pixel_tree, "outer"
    )
    som = alignment.pixel_mapping
    som = som.groupby(
        [PixelAlignment.JOIN_ORDER_COLUMN_NAME, PixelAlignment.JOIN_PIXEL_COLUMN_NAME],
        group_keys=False,
    )

    ## Lots of cute comprehension is happening here.
    ## create tuple of (source order/pixel) and [array of tuples of (object order/pixel)]
    mappy = [
        (
            HealpixPixel(source_name[0], source_name[1]),
            [
                HealpixPixel(object_elem[0], object_elem[1])
                for object_elem in object_group.dropna().to_numpy().T[:2].T
            ],
        )
        for source_name, object_group in som
    ]

    ## Treat the array of tuples as a dictionary.
    return dict(mappy)


def count_joins(args, source_order_pixel, object_order_pixels):
    """Count the number of equijoined source in the object pixels.

    If any un-joined source pixels remain, stretch out to neighboring object pixels.
    """
    source_path = pixel_catalog_file(
        catalog_base_dir=args.source_catalog_dir,
        pixel_order=source_order_pixel[0],
        pixel_number=source_order_pixel[1],
    )
    source_data = pd.read_parquet(
        path=source_path, columns=[args.source_object_id_column]
    ).set_index(args.source_object_id_column)
    remaining_sources = len(source_data)

    for object_pixel in object_order_pixels:
        object_path = pixel_catalog_file(
            catalog_base_dir=args.object_catalog_dir,
            pixel_order=object_pixel[0],
            pixel_number=object_pixel[1],
        )
        object_data = pd.read_parquet(
            path=object_path, columns=[args.object_id_column]
        ).set_index(args.object_id_column)

        joined_data = source_data.merge(
            object_data, how="inner", left_index=True, right_index=True
        )

        remaining_sources -= len(joined_data)

    if remaining_sources > 0:
        print(f"ruh-roh george. {source_order_pixel}")
