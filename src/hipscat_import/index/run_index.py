"""Create columnar index of hipscat table using dask for parallelization"""

from hipscat.io import file_io, write_metadata
from tqdm import tqdm

import hipscat_import.index.map_reduce as mr
from hipscat_import.index.arguments import IndexArguments


def _validate_args(args):
    if not args:
        raise TypeError("args is required and should be type IndexArguments")
    if not isinstance(args, IndexArguments):
        raise TypeError("args must be type IndexArguments")


def run(args):
    """Importer, where the client context may out-live the runner"""
    _validate_args(args)
    rows_written = mr.create_index(args)

    # All done - write out the metadata
    with tqdm(
        total=4, desc="Finishing", disable=not args.progress_bar
    ) as step_progress:
        catalog_params = args.to_catalog_parameters()
        catalog_params.total_rows = int(rows_written)
        write_metadata.write_provenance_info(catalog_params, args.provenance_info())
        step_progress.update(1)
        write_metadata.write_catalog_info(catalog_params)
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
        write_metadata.write_parquet_metadata(args.catalog_path)
        step_progress.update(1)
