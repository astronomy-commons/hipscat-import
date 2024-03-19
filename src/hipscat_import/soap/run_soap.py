"""Create pixel-to-pixel association between object and source catalogs.
Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from hipscat.catalog.association_catalog.partition_join_info import PartitionJoinInfo
from hipscat.io import parquet_metadata, paths, write_metadata
from tqdm.auto import tqdm

from hipscat_import.pipeline_resume_plan import PipelineResumePlan
from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import combine_partial_results, count_joins, reduce_joins
from hipscat_import.soap.resume_plan import SoapPlan


def run(args, client):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type SoapArguments")
    if not isinstance(args, SoapArguments):
        raise TypeError("args must be type SoapArguments")

    resume_plan = SoapPlan(args)
    if not resume_plan.is_counting_done():
        futures = []
        for source_pixel, object_pixels, _ in resume_plan.count_keys:
            futures.append(
                client.submit(
                    count_joins,
                    soap_args=args,
                    source_pixel=source_pixel,
                    object_pixels=object_pixels,
                )
            )

        resume_plan.wait_for_counting(futures)

    if args.write_leaf_files and not resume_plan.is_reducing_done():
        for object_pixel, object_key in resume_plan.reduce_keys:
            futures.append(
                client.submit(
                    reduce_joins,
                    soap_args=args,
                    object_pixel=object_pixel,
                    object_key=object_key,
                )
            )

        resume_plan.wait_for_reducing(futures)

    # All done - write out the metadata
    with tqdm(
        total=4, desc=PipelineResumePlan.get_formatted_stage_name("Finishing"), disable=not args.progress_bar
    ) as step_progress:
        if args.write_leaf_files:
            parquet_metadata.write_parquet_metadata(
                args.catalog_path,
                storage_options=args.output_storage_options,
            )
            total_rows = 0
            metadata_path = paths.get_parquet_metadata_pointer(args.catalog_path)
            for row_group in parquet_metadata.read_row_group_fragments(
                metadata_path,
                storage_options=args.output_storage_options,
            ):
                total_rows += row_group.num_rows
            partition_join_info = PartitionJoinInfo.read_from_file(
                metadata_path, storage_options=args.output_storage_options
            )
            partition_join_info.write_to_csv(
                catalog_path=args.catalog_path, storage_options=args.output_storage_options
            )
        else:
            total_rows = combine_partial_results(
                args.tmp_path, args.catalog_path, args.output_storage_options
            )
        step_progress.update(1)
        total_rows = int(total_rows)
        catalog_info = args.to_catalog_info(total_rows)
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        write_metadata.write_catalog_info(
            dataset_info=catalog_info,
            catalog_base_dir=args.catalog_path,
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        resume_plan.clean_resume_files()
        step_progress.update(1)
