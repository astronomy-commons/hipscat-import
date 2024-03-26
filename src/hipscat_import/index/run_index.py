"""Create columnar index of hipscat table using dask for parallelization"""

from hipscat.io import file_io, parquet_metadata, write_metadata
from tqdm.auto import tqdm

import hipscat_import.index.map_reduce as mr
from hipscat_import.index.arguments import IndexArguments
from hipscat_import.pipeline_resume_plan import PipelineResumePlan


def run(args, client):
    """Run index creation pipeline."""
    if not args:
        raise TypeError("args is required and should be type IndexArguments")
    if not isinstance(args, IndexArguments):
        raise TypeError("args must be type IndexArguments")
    rows_written = mr.create_index(args, client)

    # All done - write out the metadata
    with tqdm(
        total=4, desc=PipelineResumePlan.get_formatted_stage_name("Finishing"), disable=not args.progress_bar
    ) as step_progress:
        index_catalog_info = args.to_catalog_info(int(rows_written))
        write_metadata.write_provenance_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=index_catalog_info,
            tool_args=args.provenance_info(),
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        write_metadata.write_catalog_info(
            catalog_base_dir=args.catalog_path,
            dataset_info=index_catalog_info,
            storage_options=args.output_storage_options,
        )
        step_progress.update(1)
        file_io.remove_directory(
            args.tmp_path, ignore_errors=True, storage_options=args.output_storage_options
        )
        step_progress.update(1)
        parquet_metadata.write_parquet_metadata(
            args.catalog_path, order_by_healpix=False, storage_options=args.output_storage_options
        )
        step_progress.update(1)
