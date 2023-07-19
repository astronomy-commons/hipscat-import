"""Create pixel-to-pixel association between object and source catalogs.

Methods in this file set up a dask pipeline using futures. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from hipscat.catalog import Catalog
from hipscat.io import file_io, write_metadata
from hipscat.pixel_tree.pixel_alignment import align_trees
from hipscat.pixel_tree.pixel_tree import PixelTree
from tqdm import tqdm

from hipscat_import.soap.arguments import SoapArguments
from hipscat_import.soap.map_reduce import count_joins, source_to_object_map


def run(args, client):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type SoapArguments")
    if not isinstance(args, SoapArguments):
        raise TypeError("args must be type SoapArguments")

    with tqdm(total=1, desc="Planning", disable=not args.progress_bar) as step_progress:
        source_to_object_map(args)
        step_progress.update(1)

    with tqdm(total=1, desc="Mapping ", disable=not args.progress_bar) as step_progress:
        count_joins(args)
        step_progress.update(1)

    # All done - write out the metadata
    with tqdm(
        total=4, desc="Finishing", disable=not args.progress_bar
    ) as step_progress:
        # pylint: disable=duplicate-code
        # Very similar to /association/run_association.py
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
        ## TODO - write out join info file
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
