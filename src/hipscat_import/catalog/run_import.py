"""Import a set of non-hipscat files using dask for parallelization

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""


import hipscat.io.write_metadata as io
import numpy as np
from dask.distributed import as_completed
from hipscat import pixel_math
from tqdm import tqdm

import hipscat_import.catalog.map_reduce as mr
import hipscat_import.catalog.resume_files as resume
from hipscat_import.catalog.arguments import ImportArguments


def _map_pixels(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    raw_histogram = resume.read_histogram(args.tmp_path, args.mapping_healpix_order)
    if resume.is_mapping_done(args.tmp_path):
        return raw_histogram

    mapped_paths = set(resume.read_mapping_keys(args.tmp_path))

    futures = []
    for file_path in args.input_paths:
        map_key = f"map_{file_path}"
        if map_key in mapped_paths:
            continue
        futures.append(
            client.submit(
                mr.map_to_pixels,
                key=map_key,
                input_file=file_path,
                file_reader=args.file_reader,
                filter_function=args.filter_function,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
            )
        )

    some_error = False
    for future, result in tqdm(
        as_completed(futures, with_results=True),
        desc="Mapping  ",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        if future.status == "error":  # pragma: no cover
            some_error = True
            continue
        raw_histogram = np.add(raw_histogram, result)
        resume.write_mapping_start_key(args.tmp_path, future.key)
        resume.write_histogram(args.tmp_path, raw_histogram)
        resume.write_mapping_done_key(args.tmp_path, future.key)
    if some_error:  # pragma: no cover
        raise RuntimeError("Some mapping stages failed. See logs for details.")
    resume.set_mapping_done(args.tmp_path)
    return raw_histogram


def _split_pixels(args, alignment_future, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if resume.is_splitting_done(args.tmp_path):
        return

    split_paths = set(resume.read_splitting_keys(args.tmp_path))

    futures = []
    for i, file_path in enumerate(args.input_paths):
        split_key = f"split_{file_path}"
        if split_key in split_paths:
            continue
        futures.append(
            client.submit(
                mr.split_pixels,
                key=split_key,
                input_file=file_path,
                file_reader=args.file_reader,
                filter_function=args.filter_function,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                shard_suffix=i,
                cache_path=args.tmp_path,
                alignment=alignment_future,
            )
        )

    some_error = False
    for future in tqdm(
        as_completed(futures),
        desc="Splitting",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        if future.status == "error":  # pragma: no cover
            some_error = True
            continue
        resume.write_splitting_done_key(args.tmp_path, future.key)
    if some_error:  # pragma: no cover
        raise RuntimeError("Some splitting stages failed. See logs for details.")
    resume.set_splitting_done(args.tmp_path)


def _reduce_pixels(args, destination_pixel_map, client):
    """Loop over destination pixels and merge into parquet files"""

    if resume.is_reducing_done(args.tmp_path):
        return

    reduced_keys = set(resume.read_reducing_keys(args.tmp_path))

    futures = []
    for destination_pixel, source_pixels in destination_pixel_map.items():
        destination_pixel_key = f"{destination_pixel.order}_{destination_pixel.pixel}"
        if destination_pixel_key in reduced_keys:
            continue
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

    some_error = False
    for future in tqdm(
        as_completed(futures),
        desc="Reducing ",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        if future.status == "error":  # pragma: no cover
            some_error = True
            continue
        resume.write_reducing_key(args.tmp_path, future.key)
    if some_error:  # pragma: no cover
        raise RuntimeError("Some reducing stages failed. See logs for details.")
    resume.set_reducing_done(args.tmp_path)


def run(args, client):
    """Run catalog creation pipeline."""
    if not args:
        raise ValueError("args is required and should be type ImportArguments")
    if not isinstance(args, ImportArguments):
        raise ValueError("args must be type ImportArguments")
    raw_histogram = _map_pixels(args, client)

    with tqdm(
        total=1, desc="Binning  ", disable=not args.progress_bar
    ) as step_progress:
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
    with tqdm(
        total=6, desc="Finishing", disable=not args.progress_bar
    ) as step_progress:
        catalog_info = args.to_catalog_info(int(raw_histogram.sum()))
        io.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
        )
        step_progress.update(1)

        io.write_catalog_info(
            catalog_base_dir=args.catalog_path, dataset_info=catalog_info
        )
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
        resume.clean_resume_files(args.tmp_path)
        step_progress.update(1)
