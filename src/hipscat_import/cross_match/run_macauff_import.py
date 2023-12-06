import healpy as hp
import numpy as np
from hipscat.catalog import Catalog
from hipscat.io import file_io, parquet_metadata, paths, write_metadata
from tqdm import tqdm

import hipscat_import.catalog.map_reduce as catalog_mr
from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat_import.cross_match.macauff_map_reduce import reduce_associations

# pylint: disable=unused-argument


def split_associations(args, left_catalog):
    """Split association rows by their aligned left pixel."""
    left_pixels = left_catalog.partition_info.get_healpix_pixels()
    highest_order = left_catalog.partition_info.get_highest_order()

    regenerated_alignment = np.full(hp.order2npix(highest_order), None)
    for pixel in left_pixels:
        explosion_factor = 4 ** (highest_order - pixel.order)
        exploded_pixels = np.arange(
            pixel.pixel * explosion_factor,
            (pixel.pixel + 1) * explosion_factor,
        )
        for explody in exploded_pixels:
            regenerated_alignment[explody] = (pixel.order, pixel.pixel, 0)

    for i, file in enumerate(args.input_paths):
        catalog_mr.split_pixels(
            input_file=file,
            file_reader=args.file_reader,
            splitting_key=i,
            highest_order=highest_order,
            ra_column=args.left_ra_column,
            dec_column=args.left_dec_column,
            cache_shard_path=args.tmp_path,
            resume_path=args.tmp_path,
            alignment=regenerated_alignment,
            use_hipscat_index=False,
        )


def reduce(args, left_catalog, right_catalog):
    """Reduce left pixel files into a single parquet file per."""
    highest_right_order = right_catalog.partition_info.get_highest_order()

    left_pixels = left_catalog.partition_info.get_healpix_pixels()
    right_pixels = right_catalog.partition_info.get_healpix_pixels()

    regenerated_right_alignment = np.full(hp.order2npix(highest_right_order), None)
    for pixel in right_pixels:
        explosion_factor = 4 ** (highest_right_order - pixel.order)
        exploded_pixels = np.arange(
            pixel.pixel * explosion_factor,
            (pixel.pixel + 1) * explosion_factor,
        )
        for explody in exploded_pixels:
            regenerated_right_alignment[explody] = (pixel.order, pixel.pixel, 0)

    for left_pixel in left_pixels:
        reduce_associations(args, left_pixel, highest_right_order, regenerated_right_alignment)


def run(args, client):
    """run macauff cross-match import pipeline"""
    if not args:
        raise TypeError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise TypeError("args must be type MacauffArguments")

    left_catalog = Catalog.read_from_hipscat(args.left_catalog_dir)
    right_catalog = Catalog.read_from_hipscat(args.right_catalog_dir)

    split_associations(args, left_catalog)
    reduce(args, left_catalog, right_catalog)

    # All done - write out the metadata
    with tqdm(total=4, desc="Finishing", disable=not args.progress_bar) as step_progress:
        parquet_metadata.write_parquet_metadata(args.catalog_path)
        total_rows = 0
        metadata_path = paths.get_parquet_metadata_pointer(args.catalog_path)
        for row_group in parquet_metadata.read_row_group_fragments(metadata_path):
            total_rows += row_group.num_rows
        # pylint: disable=duplicate-code
        # Very similar to /index/run_index.py
        step_progress.update(1)
        total_rows = int(total_rows)
        catalog_info = args.to_catalog_info(total_rows)
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
        )
        step_progress.update(1)
        write_metadata.write_catalog_info(dataset_info=catalog_info, catalog_base_dir=args.catalog_path)
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
