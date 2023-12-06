import healpy as hp
import numpy as np
from hipscat.catalog import Catalog
from hipscat.io import file_io, parquet_metadata, paths, write_metadata
from tqdm import tqdm

from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat_import.cross_match.macauff_map_reduce import reduce_associations, split_associations

# pylint: disable=unused-argument


def split(args, left_catalog, right_catalog):
    """Split association rows by their aligned left pixel."""
    left_pixels = left_catalog.partition_info.get_healpix_pixels()
    highest_left_order = left_catalog.partition_info.get_highest_order()
    highest_right_order = right_catalog.partition_info.get_highest_order()

    left_pixels = left_catalog.partition_info.get_healpix_pixels()
    right_pixels = right_catalog.partition_info.get_healpix_pixels()
    regenerated_left_alignment = np.full(hp.order2npix(highest_left_order), None)
    for pixel in left_pixels:
        explosion_factor = 4 ** (highest_left_order - pixel.order)
        exploded_pixels = np.arange(
            pixel.pixel * explosion_factor,
            (pixel.pixel + 1) * explosion_factor,
        )
        regenerated_left_alignment[exploded_pixels] = pixel

    regenerated_right_alignment = np.full(hp.order2npix(highest_right_order), None)
    for pixel in right_pixels:
        explosion_factor = 4 ** (highest_right_order - pixel.order)
        exploded_pixels = np.arange(
            pixel.pixel * explosion_factor,
            (pixel.pixel + 1) * explosion_factor,
        )
        regenerated_right_alignment[exploded_pixels] = pixel

    for i, file in enumerate(args.input_paths):
        split_associations(
            input_file=file,
            file_reader=args.file_reader,
            splitting_key=i,
            args=args,
            highest_left_order=highest_left_order,
            highest_right_order=highest_right_order,
            left_alignment=regenerated_left_alignment,
            right_alignment=regenerated_right_alignment,
        )


def reduce(args, left_catalog):
    """Reduce left pixel files into a single parquet file per."""
    left_pixels = left_catalog.partition_info.get_healpix_pixels()

    for left_pixel in left_pixels:
        reduce_associations(args, left_pixel)


def run(args, client):
    """run macauff cross-match import pipeline"""
    if not args:
        raise TypeError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise TypeError("args must be type MacauffArguments")

    left_catalog = Catalog.read_from_hipscat(args.left_catalog_dir)
    right_catalog = Catalog.read_from_hipscat(args.right_catalog_dir)

    split(args, left_catalog, right_catalog)
    reduce(args, left_catalog)

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
