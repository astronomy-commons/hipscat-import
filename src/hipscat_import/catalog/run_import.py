"""Import a set of non-hipscat files using dask for parallelization

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""


import hipscat.io.write_metadata as io
import numpy as np
from hipscat import pixel_math
from tqdm import tqdm

import hipscat_import.catalog.map_reduce as mr
from hipscat_import.catalog.arguments import ImportArguments


def _map_pixels(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if args.resume_plan.is_mapping_done():
        return

    reader_future = client.scatter(args.file_reader)
    futures = []
    for key, file_path in args.resume_plan.map_files:
        futures.append(
            client.submit(
                mr.map_to_pixels,
                key=key,
                input_file=file_path,
                cache_path=args.tmp_path,
                file_reader=reader_future,
                mapping_key=key,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
            )
        )
    args.resume_plan.wait_for_mapping(futures)


def _split_pixels(args, alignment_future, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if args.resume_plan.is_splitting_done():
        return

    reader_future = client.scatter(args.file_reader)
    futures = []
    for key, file_path in args.resume_plan.split_keys:
        futures.append(
            client.submit(
                mr.split_pixels,
                key=key,
                input_file=file_path,
                file_reader=reader_future,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                shard_suffix=key,
                cache_path=args.tmp_path,
                alignment=alignment_future,
            )
        )

    args.resume_plan.wait_for_splitting(futures)


def _reduce_pixels(args, destination_pixel_map, client):
    """Loop over destination pixels and merge into parquet files"""

    if args.resume_plan.is_reducing_done():
        return

    futures = []
    for (
        destination_pixel,
        source_pixels,
        destination_pixel_key,
    ) in args.resume_plan.get_reduce_items(destination_pixel_map):
        futures.append(
            client.submit(
                mr.reduce_pixel_shards,
                key=destination_pixel_key,
                cache_path=args.tmp_path,
                destination_pixel_order=destination_pixel.order,
                destination_pixel_number=destination_pixel.pixel,
                destination_pixel_size=source_pixels[0],
                output_path=args.catalog_path,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                id_column=args.id_column,
                add_hipscat_index=args.add_hipscat_index,
                use_schema_file=args.use_schema_file,
            )
        )

    args.resume_plan.wait_for_reducing(futures)


def run(args, client):
    """Run catalog creation pipeline."""
    if not args:
        raise ValueError("args is required and should be type ImportArguments")
    if not isinstance(args, ImportArguments):
        raise ValueError("args must be type ImportArguments")
    _map_pixels(args, client)

    with tqdm(total=2, desc="Binning  ", disable=not args.progress_bar) as step_progress:
        raw_histogram = args.resume_plan.read_histogram(args.mapping_healpix_order)
        step_progress.update(1)
        if args.constant_healpix_order >= 0:
            alignment = np.full(len(raw_histogram), None)
            for pixel_num, pixel_sum in enumerate(raw_histogram):
                alignment[pixel_num] = (
                    args.constant_healpix_order,
                    pixel_num,
                    pixel_sum,
                )

            destination_pixel_map = pixel_math.generate_constant_pixel_map(
                histogram=raw_histogram,
                constant_healpix_order=args.constant_healpix_order,
            )
        else:
            alignment = pixel_math.generate_alignment(
                raw_histogram,
                highest_order=args.highest_healpix_order,
                threshold=args.pixel_threshold,
            )
            destination_pixel_map = pixel_math.compute_pixel_map(
                raw_histogram,
                highest_order=args.highest_healpix_order,
                threshold=args.pixel_threshold,
            )
        step_progress.update(1)

    if not args.debug_stats_only:
        alignment_future = client.scatter(alignment)
        _split_pixels(args, alignment_future, client)
        _reduce_pixels(args, destination_pixel_map, client)

    # All done - write out the metadata
    with tqdm(total=6, desc="Finishing", disable=not args.progress_bar) as step_progress:
        catalog_info = args.to_catalog_info(int(raw_histogram.sum()))
        io.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
        )
        step_progress.update(1)

        io.write_catalog_info(catalog_base_dir=args.catalog_path, dataset_info=catalog_info)
        step_progress.update(1)
        if not args.debug_stats_only:
            io.write_parquet_metadata(args.catalog_path)
        step_progress.update(1)
        io.write_fits_map(args.catalog_path, raw_histogram)
        step_progress.update(1)
        io.write_partition_info(
            catalog_base_dir=args.catalog_path,
            destination_healpix_pixel_map=destination_pixel_map,
        )
        step_progress.update(1)
        args.resume_plan.clean_resume_files()
        step_progress.update(1)
