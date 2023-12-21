import healpy as hp
import numpy as np
from hipscat.catalog import Catalog
from hipscat.io import file_io, parquet_metadata, paths, write_metadata
from tqdm import tqdm

from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat_import.cross_match.macauff_map_reduce import reduce_associations, split_associations
from hipscat_import.cross_match.macauff_resume_plan import MacauffResumePlan

# pylint: disable=unused-argument


def split(
    args,
    highest_left_order,
    highest_right_order,
    left_alignment,
    right_alignment,
    resume_plan,
    client,
):
    """Split association rows by their aligned left pixel."""

    if resume_plan.is_splitting_done():
        return

    reader_future = client.scatter(args.file_reader)
    left_alignment_future = client.scatter(left_alignment)
    right_alignment_future = client.scatter(right_alignment)
    futures = []
    for key, file_path in resume_plan.split_keys:
        futures.append(
            client.submit(
                split_associations,
                input_file=file_path,
                file_reader=reader_future,
                splitting_key=key,
                highest_left_order=highest_left_order,
                highest_right_order=highest_right_order,
                left_alignment=left_alignment_future,
                right_alignment=right_alignment_future,
                left_ra_column=args.left_ra_column,
                left_dec_column=args.left_dec_column,
                right_ra_column=args.right_ra_column,
                right_dec_column=args.right_dec_column,
                tmp_path=args.tmp_path,
            )
        )
    resume_plan.wait_for_splitting(futures)


def reduce(args, left_pixels, resume_plan, client):
    """Reduce left pixel files into a single parquet file per."""

    if resume_plan.is_reducing_done():
        return

    futures = []
    for (
        left_pixel,
        pixel_key,
    ) in resume_plan.reduce_keys:
        futures.append(
            client.submit(
                reduce_associations,
                left_pixel=left_pixel,
                tmp_path=args.tmp_path,
                catalog_path=args.catalog_path,
                reduce_key=pixel_key,
            )
        )

    resume_plan.wait_for_reducing(futures, left_pixels)


def run(args, client):
    """run macauff cross-match import pipeline"""
    if not args:
        raise TypeError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise TypeError("args must be type MacauffArguments")

    ## Lump all of the catalog reading stuff together under a single block,
    ## since it can take a while to load large catalogs and some feedback is nice.
    with tqdm(total=5, desc="Planning", disable=not args.progress_bar) as step_progress:
        left_catalog = Catalog.read_from_hipscat(args.left_catalog_dir)
        step_progress.update(1)
        right_catalog = Catalog.read_from_hipscat(args.right_catalog_dir)
        step_progress.update(1)
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
        step_progress.update(1)

        regenerated_right_alignment = np.full(hp.order2npix(highest_right_order), None)
        for pixel in right_pixels:
            explosion_factor = 4 ** (highest_right_order - pixel.order)
            exploded_pixels = np.arange(
                pixel.pixel * explosion_factor,
                (pixel.pixel + 1) * explosion_factor,
            )
            regenerated_right_alignment[exploded_pixels] = pixel
        step_progress.update(1)

        resume_plan = MacauffResumePlan(args, left_pixels)
        step_progress.update(1)

    split(
        args,
        highest_left_order=highest_left_order,
        highest_right_order=highest_right_order,
        left_alignment=regenerated_left_alignment,
        right_alignment=regenerated_right_alignment,
        resume_plan=resume_plan,
        client=client,
    )
    reduce(args, left_pixels, resume_plan, client)

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
