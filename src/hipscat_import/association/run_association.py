"""Create partitioned association table between two catalogs
using dask dataframes for parallelization

Methods in this file set up a dask pipeline using dataframes. 
The actual logic of the map reduce is in the `map_reduce.py` file.
"""

from hipscat.io import file_io, write_metadata
from tqdm import tqdm

from hipscat_import.association.arguments import AssociationArguments
from hipscat_import.association.map_reduce import map_association, reduce_association


def run(args):
    """Run the association pipeline"""
    if not args:
        raise TypeError("args is required and should be type AssociationArguments")
    if not isinstance(args, AssociationArguments):
        raise TypeError("args must be type AssociationArguments")

    with tqdm(total=1, desc="Mapping ", disable=not args.progress_bar) as step_progress:
        map_association(args)
        step_progress.update(1)

    rows_written = 0
    with tqdm(
        total=1, desc="Reducing ", disable=not args.progress_bar
    ) as step_progress:
        rows_written = reduce_association(args.tmp_path, args.catalog_path)
        step_progress.update(1)

    # All done - write out the metadata
    with tqdm(
        total=4, desc="Finishing", disable=not args.progress_bar
    ) as step_progress:
        # pylint: disable=duplicate-code
        # Very similar to /index/run_index.py
        catalog_info = args.to_catalog_info(int(rows_written))
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=catalog_info,
            tool_args=args.provenance_info(),
        )
        step_progress.update(1)
        catalog_info = args.to_catalog_info(total_rows=int(rows_written))
        write_metadata.write_catalog_info(
            dataset_info=catalog_info, catalog_base_dir=args.catalog_path
        )
        step_progress.update(1)
        write_metadata.write_parquet_metadata(args.catalog_path)
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
