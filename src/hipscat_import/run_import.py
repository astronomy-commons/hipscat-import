"""Import a set of non-hipscat files using dask for parallelization

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""


import hipscat.io.write_metadata as io
import numpy as np
from dask.distributed import Client, as_completed
from hipscat import pixel_math
from tqdm import tqdm

import hipscat_import.map_reduce as mr
import hipscat_import.resume_files as resume
from hipscat_import.arguments import ImportArguments


def _map_pixels(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    raw_histogram = resume.read_histogram(args.tmp_path, args.highest_healpix_order)
    if resume.is_mapping_done(args.tmp_path):
        return raw_histogram

    mapped_paths = set(resume.read_mapping_keys(args.tmp_path))

    futures = []
    for i, file_path in enumerate(args.input_paths):
        if file_path in mapped_paths:
            continue
        futures.append(
            client.submit(
                mr.map_to_pixels,
                key=file_path,
                input_file=file_path,
                file_reader=args.file_reader,
                filter_function=args.filter_function,
                highest_order=args.highest_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                shard_suffix=i,
                cache_path=None if args.debug_stats_only else args.tmp_path,
            )
        )

    for future, result in tqdm(
        as_completed(futures, with_results=True),
        desc="Mapping  ",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        raw_histogram = np.add(raw_histogram, result)
        resume.write_mapping_start_key(args.tmp_path, future.key)
        resume.write_histogram(args.tmp_path, raw_histogram)
        resume.write_mapping_done_key(args.tmp_path, future.key)
    resume.set_mapping_done(args.tmp_path)
    return raw_histogram


def _reduce_pixels(args, destination_pixel_map, client):
    """Loop over destination pixels and merge into parquet files"""

    if resume.is_reducing_done(args.tmp_path):
        return

    reduced_keys = set(resume.read_reducing_keys(args.tmp_path))

    futures = []
    for destination_pixel, source_pixels in destination_pixel_map.items():
        destination_pixel_key = f"{destination_pixel[0]}_{destination_pixel[1]}"
        if destination_pixel_key in reduced_keys:
            continue
        futures.append(
            client.submit(
                mr.reduce_pixel_shards,
                key=destination_pixel_key,
                cache_path=args.tmp_path,
                origin_pixel_numbers=source_pixels,
                destination_pixel_order=destination_pixel[0],
                destination_pixel_number=destination_pixel[1],
                destination_pixel_size=destination_pixel[2],
                output_path=args.catalog_path,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                id_column=args.id_column,
                add_hipscat_index=args.add_hipscat_index,
            )
        )
    for future in tqdm(
        as_completed(futures),
        desc="Reducing ",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        resume.write_reducing_key(args.tmp_path, future.key)
    resume.set_reducing_done(args.tmp_path)


def _validate_args(args):
    if not args:
        raise ValueError("args is required and should be type ImportArguments")
    if not isinstance(args, ImportArguments):
        raise ValueError("args must be type ImportArguments")


def run(args):
    """Importer that creates a dask client from the arguments"""
    _validate_args(args)

    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:  # pragma: no cover
        run_with_client(args, client)


def run_with_client(args, client):
    """Importer, where the client context may out-live the runner"""
    _validate_args(args)
    raw_histogram = _map_pixels(args, client)

    step_progress = tqdm(total=2, desc="Binning  ", disable=not args.progress_bar)
    pixel_map = pixel_math.generate_alignment(
        raw_histogram, args.highest_healpix_order, args.pixel_threshold
    )
    step_progress.update(1)
    destination_pixel_map = pixel_math.generate_destination_pixel_map(
        raw_histogram, pixel_map
    )
    step_progress.update(1)
    step_progress.close()

    if not args.debug_stats_only:
        _reduce_pixels(args, destination_pixel_map, client)

    # All done - write out the metadata
    step_progress = tqdm(total=6, desc="Finishing", disable=not args.progress_bar)
    io.write_provenance_info(args.to_catalog_parameters(), args.provenance_info())
    step_progress.update(1)
    io.write_catalog_info(args.to_catalog_parameters(), raw_histogram)
    step_progress.update(1)
    if not args.debug_stats_only:
        io.write_parquet_metadata(args.catalog_path)
    step_progress.update(1)
    io.write_fits_map(args.catalog_path, raw_histogram)
    step_progress.update(1)
    io.write_partition_info(args.to_catalog_parameters(), destination_pixel_map)
    step_progress.update(1)
    resume.clean_resume_files(args.tmp_path)
    step_progress.update(1)
    step_progress.close()
