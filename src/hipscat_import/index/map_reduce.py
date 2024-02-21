"""Create columnar index of hipscat table using dask for parallelization"""

import dask.dataframe as dd
import numpy as np
from dask.distributed import progress, wait
from hipscat.io import paths
from hipscat.io.file_io import file_io
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN


def create_index(args, client):
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
        ## Take out the hive dictionary behavior that turns these into int32.
        data["Norder"] = data["Norder"].astype(np.uint8)
        data["Dir"] = data["Dir"].astype(np.uint64)
        data["Npix"] = data["Npix"].astype(np.uint64)

    # There are some silly dask things happening here:
    # - Turn the existing index column into a regular column
    # - If that had been the _hipscat_index, and we don't want it anymore, drop it
    # - Create a new index, using our target indexing_column.
    #   Use division hints if provided.
    data = data.reset_index()
    if not args.include_hipscat_index:
        data = data.drop(columns=[HIPSCAT_ID_COLUMN])

    if args.division_hints is not None and len(args.division_hints) > 2:
        data = data.set_index(args.indexing_column, divisions=args.division_hints)
    else:
        # Try to avoid this! It's expensive! See:
        # https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.set_index.html
        data = data.set_index(args.indexing_column)

    if args.drop_duplicates:
        # More dask things:
        # - Repartition the whole dataset to account for limited memory in
        #   pyarrow in the drop_duplicates implementation (
        #   "array cannot contain more than 2147483646 bytes")
        # - Reset the index, so the indexing_column values can be considered
        #   when de-duping.
        # - Drop duplicate rows
        # - Set the index back to our indexing_column, but this time, the
        #   values are still sorted so it's cheaper.
        data = (
            data.repartition(partition_size=1_000_000_000)
            .reset_index()
            .drop_duplicates()
            .set_index(args.indexing_column, sorted=True, partition_size=args.compute_partition_size)
        )
    else:
        data = data.repartition(partition_size=args.compute_partition_size)

    # Now just write it out to leaf parquet files!
    result = data.to_parquet(
        path=index_dir,
        engine="pyarrow",
        compute_kwargs={"partition_size": args.compute_partition_size},
    )
    client.compute(result)
    return len(data)
