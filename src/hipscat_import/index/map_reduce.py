"""Create columnar index of hipscat table using dask for parallelization"""

import dask.dataframe as dd
import numpy as np
from dask.distributed import progress, wait
from hipscat.io import paths
from hipscat.io.file_io import file_io
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN


def create_index(args):
    """Read primary column, indexing column, and other payload data,
    and write to catalog directory."""
    include_columns = [args.indexing_column]
    if args.extra_columns:
        include_columns.extend(args.extra_columns)
    if args.include_order_pixel:
        include_columns.extend(["Norder", "Dir", "Npix"])

    index_dir = paths.append_paths_to_pointer(args.catalog_path, "index")

    metadata_file = paths.get_parquet_metadata_pointer(args.input_catalog_path)

    metadata = file_io.read_parquet_metadata(metadata_file)
    data = dd.read_parquet(
        path=args.input_catalog_path,
        columns=include_columns,
        engine="pyarrow",
        dataset={"partitioning": {"flavor": "hive", "schema": metadata.schema.to_arrow_schema()}},
        filesystem="arrow",
    )

    if args.include_order_pixel:
        ## Take out the hive dictionary behavior.
        data["Norder"] = data["Norder"].astype(np.uint8)
        data["Dir"] = data["Dir"].astype(np.uint64)
        data["Npix"] = data["Npix"].astype(np.uint64)
    data = data.reset_index()
    if not args.include_hipscat_index:
        data = data.drop(columns=[HIPSCAT_ID_COLUMN])
    data = data.drop_duplicates()
    data = data.repartition(partition_size=args.compute_partition_size)
    data = data.set_index(args.indexing_column)
    result = data.to_parquet(
        path=index_dir,
        engine="pyarrow",
        compute_kwargs={"partition_size": args.compute_partition_size},
    )
    if args.progress_bar:  # pragma: no cover
        progress(result)
    else:
        wait(result)
    return len(data)
