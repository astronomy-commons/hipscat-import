"""Import a set of non-hipscat files using dask for parallelization

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

import hipscat.io.write_metadata as io
import numpy as np
from hipscat import pixel_math
from hipscat.catalog import PartitionInfo
from hipscat.io import paths
from hipscat.io.parquet_metadata import write_parquet_metadata
from tqdm.auto import tqdm

import hipscat_import.catalog.map_reduce as mr
from hipscat_import.catalog.arguments import ImportArguments
from hipscat_import.pipeline_resume_plan import PipelineResumePlan


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
                input_file=file_path,
                resume_path=args.resume_plan.tmp_path,
                file_reader=reader_future,
                mapping_key=key,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                use_hipscat_index=args.use_hipscat_index,
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
                input_file=file_path,
                file_reader=reader_future,
                highest_order=args.mapping_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                splitting_key=key,
                cache_shard_path=args.tmp_path,
                resume_path=args.resume_plan.tmp_path,
                alignment=alignment_future,
                use_hipscat_index=args.use_hipscat_index,
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
                cache_shard_path=args.tmp_path,
                resume_path=args.resume_plan.tmp_path,
                reducing_key=destination_pixel_key,
                destination_pixel_order=destination_pixel.order,
                destination_pixel_number=destination_pixel.pixel,
                destination_pixel_size=source_pixels[0],
                output_path=args.catalog_path,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                sort_columns=args.sort_columns,
                add_hipscat_index=args.add_hipscat_index,
                use_schema_file=args.use_schema_file,
                use_hipscat_index=args.use_hipscat_index,
                storage_options=args.output_storage_options,
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

    with tqdm(
        total=2, desc=PipelineResumePlan.get_formatted_stage_name("Binning"), disable=not args.progress_bar
    ) as step_progress:
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
                lowest_order=args.lowest_healpix_order,
                threshold=args.pixel_threshold,
            )
            destination_pixel_map = pixel_math.compute_pixel_map(
                raw_histogram,
                highest_order=args.highest_healpix_order,
                lowest_order=args.lowest_healpix_order,
                threshold=args.pixel_threshold,
            )
        step_progress.update(1)

    if not args.debug_stats_only:
        alignment_future = client.scatter(alignment)
        _split_pixels(args, alignment_future, client)
        _reduce_pixels(args, destination_pixel_map, client)

    # All done - write out the metadata
    with tqdm(
        total=5, desc=PipelineResumePlan.get_formatted_stage_name("Finishing"), disable=not args.progress_bar
    ) as step_progress:
        catalog_info = args.to_catalog_info(int(raw_histogram.sum()))
        io.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)

        io.write_catalog_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        partition_info = PartitionInfo.from_healpix(destination_pixel_map.keys())
        partition_info_file = paths.get_partition_info_pointer(args.catalog_path)
        partition_info.write_to_file(partition_info_file, storage_options=args.output_storage_options)
        if not args.debug_stats_only:
            write_parquet_metadata(args.catalog_path, storage_options=args.output_storage_options)
        else:
            partition_info.write_to_metadata_files(
                args.catalog_path, storage_options=args.output_storage_options
            )
        step_progress.update(1)
        io.write_fits_map(args.catalog_path, raw_histogram, storage_options=args.output_storage_options)
        step_progress.update(1)
        args.resume_plan.clean_resume_files()
        step_progress.update(1)
