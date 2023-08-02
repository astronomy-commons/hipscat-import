"""Create pixel-to-pixel association between object and source catalogs.
Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from hipscat.io import file_io, write_metadata
from tqdm import tqdm

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import combine_partial_results, count_joins
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
        for source_pixel, object_pixels, source_key in resume_plan.count_keys:
            futures.append(
                client.submit(
                    count_joins,
                    key=source_key,
                    soap_args=args,
                    source_pixel=source_pixel,
                    object_pixels=object_pixels,
                    cache_path=args.tmp_path,
                )
            )

        resume_plan.wait_for_counting(futures)

    # All done - write out the metadata
    with tqdm(total=4, desc="Finishing", disable=not args.progress_bar) as step_progress:
        # pylint: disable=duplicate-code
        # Very similar to /index/run_index.py
        combine_partial_results(args.tmp_path, args.catalog_path)
        step_progress.update(1)
        catalog_info = args.to_catalog_info(0)
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
