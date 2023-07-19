"""Create pixel-to-pixel association between object and source catalogs.

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from dask.distributed import as_completed
from hipscat.io import file_io, write_metadata
from tqdm import tqdm

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import (
    combine_partial_results,
    count_joins,
    source_to_object_map,
)


def run(args, client):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type SoapArguments")
    if not isinstance(args, SoapArguments):
        raise TypeError("args must be type SoapArguments")

    with tqdm(total=1, desc="Planning", disable=not args.progress_bar) as step_progress:
        source_to_object, source_to_neighbor_object = source_to_object_map(args)
        step_progress.update(1)

    futures = []
    for source, objects in source_to_object.items():
        futures.append(
            client.submit(
                count_joins,
                args,
                source,
                objects,
                source_to_neighbor_object[source],
                args.tmp_path,
            )
        )

    some_error = False
    for future in tqdm(
        as_completed(futures),
        desc="Counting ",
        total=len(futures),
        disable=(not args.progress_bar),
    ):
        if future.status == "error":  # pragma: no cover
            some_error = True
    if some_error:  # pragma: no cover
        raise RuntimeError("Some Counting stages failed. See logs for details.")

    # All done - write out the metadata
    with tqdm(
        total=4, desc="Finishing", disable=not args.progress_bar
    ) as step_progress:
        # pylint: disable=duplicate-code
        # Very similar to /association/run_association.py
        combine_partial_results(args.tmp_path, args.catalog_path)
        step_progress.update(1)
        catalog_info = args.to_catalog_info(0)
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
        )
        step_progress.update(1)
        write_metadata.write_catalog_info(
            dataset_info=catalog_info, catalog_base_dir=args.catalog_path
        )
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
