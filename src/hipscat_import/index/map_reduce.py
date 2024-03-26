"""Create columnar index of hipscat table using dask for parallelization"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
from hipscat.io import paths
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN


def read_leaf_file(input_file, include_columns, include_hipscat_index, drop_duplicates, storage_options):
    """Mapping function called once per input file.

    Reads the leaf parquet file, and returns with appropriate columns and duplicates dropped."""
    data = pd.read_parquet(
        input_file,
        columns=include_columns,
        engine="pyarrow",
        storage_options=storage_options,
    )

    data = data.reset_index()
    if not include_hipscat_index:
        data = data.drop(columns=[HIPSCAT_ID_COLUMN])

    if drop_duplicates:
        data = data.drop_duplicates()
    return data


def create_index(args, client):
    """Read primary column, indexing column, and other payload data,
    and write to catalog directory."""
    include_columns = [args.indexing_column]
    if args.extra_columns:
        include_columns.extend(args.extra_columns)
    if args.include_order_pixel:
        include_columns.extend(["Norder", "Dir", "Npix"])

    index_dir = paths.append_paths_to_pointer(args.catalog_path, "index")

    data = dd.from_map(
        read_leaf_file,
        [
            paths.pixel_catalog_file(
                catalog_base_dir=args.input_catalog.catalog_base_dir,
                pixel_order=pixel.order,
                pixel_number=pixel.pixel,
            )
            for pixel in args.input_catalog.get_healpix_pixels()
        ],
        include_columns=include_columns,
        include_hipscat_index=args.include_hipscat_index,
        drop_duplicates=args.drop_duplicates,
        storage_options=args.input_storage_options,
    )

    if args.include_order_pixel:
        ## Take out the hive dictionary behavior that turns these into int32.
        data["Norder"] = data["Norder"].astype(np.uint8)
        data["Dir"] = data["Dir"].astype(np.uint64)
        data["Npix"] = data["Npix"].astype(np.uint64)

    if args.division_hints is not None and len(args.division_hints) > 2:
        data = data.set_index(args.indexing_column, divisions=args.division_hints)
    else:
        # Try to avoid this! It's expensive! See:
        # https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.set_index.html
        data = data.set_index(args.indexing_column)

    data = data.repartition(partition_size=args.compute_partition_size)

    # Now just write it out to leaf parquet files!
    result = data.to_parquet(
        path=index_dir,
        engine="pyarrow",
        compute_kwargs={"partition_size": args.compute_partition_size},
        storage_options=args.output_storage_options,
    )
    client.compute(result)
    return len(data)
