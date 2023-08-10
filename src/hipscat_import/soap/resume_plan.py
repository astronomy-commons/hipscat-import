"""Utility to hold the pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import healpy as hp
import numpy as np
from hipscat.catalog import Catalog
from hipscat.io import file_io
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_tree import PixelAlignment, align_trees
from tqdm import tqdm

from hipscat_import.pipeline_resume_plan import PipelineResumePlan
from hipscat_import.soap.arguments import SoapArguments


@dataclass
class SoapPlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    count_keys: List[Tuple[HealpixPixel, List[HealpixPixel], str]] = field(default_factory=list)
    """set of pixels (and job keys) that have yet to be counted"""

    COUNTING_STAGE = "counting"
    SOURCE_MAP_FILE = "source_object_map.npz"

    def __init__(self, args: SoapArguments):
        if not args.tmp_path:  # pragma: no cover (not reachable, but required for mypy)
            raise ValueError("tmp_path is required")
        super().__init__(resume=args.resume, progress_bar=args.progress_bar, tmp_path=args.tmp_path)
        self.gather_plan(args)

    def gather_plan(self, args):
        """Initialize the plan."""
        with tqdm(total=5, desc="Planning ", disable=not self.progress_bar) as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            step_progress.update(1)
            self.count_keys = []

            ## Validate existing resume state.
            if self.is_counting_done():
                return
            step_progress.update(1)

            source_map_file = file_io.append_paths_to_pointer(self.tmp_path, self.SOURCE_MAP_FILE)
            if file_io.does_file_or_directory_exist(source_map_file):
                source_pixel_map = np.load(source_map_file, allow_pickle=True)["arr_0"].item()
            else:
                object_catalog = Catalog.read_from_hipscat(args.object_catalog_dir)
                source_catalog = Catalog.read_from_hipscat(args.source_catalog_dir)
                source_pixel_map = source_to_object_map(object_catalog, source_catalog)
                np.savez_compressed(source_map_file, source_pixel_map)
            self._set_sources_to_count(source_pixel_map)
            step_progress.update(1)

    def wait_for_counting(self, futures):
        """Wait for counting stage futures to complete."""
        self.wait_for_futures(futures, self.COUNTING_STAGE)

    def is_counting_done(self) -> bool:
        """Are there sources left to count?"""
        return self.done_file_exists(self.COUNTING_STAGE)

    def _set_sources_to_count(self, source_pixel_map):
        """Fetch a triple for each source pixel to join and count.

        Triple contains:
            - source pixel
            - object pixels (healpix pixel with both order and pixel, for aligning and
              neighboring object pixels)
            - source key (string of source order+pixel)

        """
        counted_keys = set(self.read_log_keys(self.COUNTING_STAGE))
        self.count_keys = [
            (hp_pixel, object_pixels, f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel, object_pixels in source_pixel_map.items()
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in counted_keys
        ]


def source_to_object_map(object_catalog, source_catalog):
    """Build a map of (source order/pixel) to the (object order/pixel)
    that are aligned, as well as neighboring object pixels.
    """

    ## Direct aligment from source to object
    ###############################################
    grouped_alignment = align_trees(
        object_catalog.pixel_tree, source_catalog.pixel_tree, "outer"
    ).pixel_mapping.groupby(
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
        for source_name, object_group in grouped_alignment
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

    for source, objects in source_to_object.items():
        # get all neighboring pixels
        nside = hp.order2nside(source.order)
        neighbors = hp.get_all_neighbours(nside, source.pixel, nest=True)

        ## get rid of -1s and normalize to max order
        explosion_factor = 4 ** (max_order - source.order)
        ## explode out the source pixels to the same order as object map
        ## NB: This may find non-bordering object neighbors, but that's ok!
        neighbors = [
            [
                *range(
                    pixel * explosion_factor,
                    (pixel + 1) * explosion_factor,
                )
            ]
            for pixel in neighbors
            if pixel != -1
        ]
        ## Flatten out the exploded list of lists
        neighbors = [item for sublist in neighbors for item in sublist]

        neighbors_orders = object_order_map[neighbors]
        desploded = [
            HealpixPixel(order, hoo_pixel >> 2 * (max_order - order))
            for order, hoo_pixel in list(zip(neighbors_orders, neighbors))
            if order != -1
        ]
        neighbors = set(desploded) - set(objects)
        objects.extend(list(neighbors))

    return source_to_object
