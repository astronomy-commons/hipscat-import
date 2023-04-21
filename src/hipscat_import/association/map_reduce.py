"""Create partitioned association table between two catalogs"""

import dask.dataframe as dd
import pyarrow.parquet as pq
from hipscat.io import file_io, paths


def map_association(args):
    """Using dask dataframes, create an association between two catalogs.
    This will write out sharded parquet files to the temp (intermediate)
    directory.

    Implementation notes:

    Because we may be joining to a column that is NOT the natural/primary key
    (on either side of the join), we fetch both the identifying column and the
    join predicate column, possibly duplicating one of the columns.

    This way, when we drop the join predicate columns at the end of the process,
    we will still have the identifying columns. However, it makes the loading
    of each input catalog more verbose.
    """
    ## Read and massage primary input data
    ## NB: We may be joining on a column that is NOT the natural primary key.
    single_primary_column = args.primary_id_column == args.primary_join_column
    read_columns = [
        "Norder",
        "Dir",
        "Npix",
    ]
    if single_primary_column:
        read_columns = [args.primary_id_column] + read_columns
    else:
        read_columns = [args.primary_join_column, args.primary_id_column] + read_columns

    primary_index = dd.read_parquet(
        path=args.primary_input_catalog_path,
        columns=read_columns,
        dataset={"partitioning": "hive"},
    )
    if single_primary_column:
        ## Duplicate the column to simplify later steps
        primary_index["primary_id"] = primary_index[args.primary_join_column]
    rename_columns = {
        args.primary_join_column: "primary_join",
        "_hipscat_index": "primary_hipscat_index",
    }
    if not single_primary_column:
        rename_columns[args.primary_id_column] = "primary_id"
    primary_index = (
        primary_index.reset_index()
        .rename(columns=rename_columns)
        .set_index("primary_join")
    )

    ## Read and massage join input data
    single_join_column = args.join_id_column == args.join_foreign_key
    read_columns = [
        "Norder",
        "Dir",
        "Npix",
    ]
    if single_join_column:
        read_columns = [args.join_id_column] + read_columns
    else:
        read_columns = [args.join_id_column, args.join_foreign_key] + read_columns

    join_index = dd.read_parquet(
        path=args.join_input_catalog_path,
        columns=read_columns,
        dataset={"partitioning": "hive"},
    )
    if single_join_column:
        ## Duplicate the column to simplify later steps
        join_index["join_id"] = join_index[args.join_id_column]
    rename_columns = {
        args.join_foreign_key: "join_to_primary",
        "_hipscat_index": "join_hipscat_index",
        "Norder": "join_Norder",
        "Dir": "join_Dir",
        "Npix": "join_Npix",
    }
    if not single_join_column:
        rename_columns[args.join_id_column] = "join_id"
    join_index = (
        join_index.reset_index()
        .rename(columns=rename_columns)
        .set_index("join_to_primary")
    )

    ## Join the two data sets on the shared join predicate.
    join_data = primary_index.merge(
        join_index, how="inner", left_index=True, right_index=True
    )

    ## Write out a summary of each partition join
    groups = (
        join_data.groupby(
            ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"],
            group_keys=False,
        )["primary_hipscat_index"]
        .count()
        .compute()
    )
    intermediate_partitions_file = file_io.append_paths_to_pointer(
        args.tmp_path, "partitions.csv"
    )
    file_io.write_dataframe_to_csv(
        dataframe=groups, file_pointer=intermediate_partitions_file
    )

    ## Drop join predicate columns
    join_data = join_data[
        [
            "Norder",
            "Dir",
            "Npix",
            "join_Norder",
            "join_Dir",
            "join_Npix",
            "primary_id",
            "primary_hipscat_index",
            "join_id",
            "join_hipscat_index",
        ]
    ]

    ## Write out association table shards.
    join_data.to_parquet(
        path=args.tmp_path,
        engine="pyarrow",
        partition_on=["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"],
        compute_kwargs={"partition_size": args.compute_partition_size},
        write_index=False,
    )


def reduce_association(input_path, output_path):
    """Collate sharded parquet files into a single parquet file per partition"""
    intermediate_partitions_file = file_io.append_paths_to_pointer(
        input_path, "partitions.csv"
    )
    data_frame = file_io.load_csv_to_pandas(intermediate_partitions_file)

    ## Clean up the dataframe and write out as our new partition join info file.
    data_frame = data_frame[data_frame["primary_hipscat_index"] != 0]
    data_frame["num_rows"] = data_frame["primary_hipscat_index"]
    data_frame = data_frame[
        ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]
    ]
    data_frame = data_frame.sort_values(["Norder", "Npix", "join_Norder", "join_Npix"])
    file_io.write_dataframe_to_csv(
        dataframe=data_frame,
        file_pointer=file_io.append_paths_to_pointer(
            output_path, "partition_join_info.csv"
        ),
        index=False,
    )

    ## For each partition, join all parquet shards into single parquet file.
    for _, partition in data_frame.iterrows():
        input_dir = paths.create_hive_directory_name(
            input_path,
            ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"],
            [
                partition["Norder"],
                partition["Dir"],
                partition["Npix"],
                partition["join_Norder"],
                partition["join_Dir"],
                partition["join_Npix"],
            ],
        )
        output_dir = paths.pixel_association_directory(
            output_path,
            partition["Norder"],
            partition["Npix"],
            partition["join_Norder"],
            partition["join_Npix"],
        )
        file_io.make_directory(output_dir, exist_ok=True)
        output_file = paths.pixel_association_file(
            output_path,
            partition["Norder"],
            partition["Npix"],
            partition["join_Norder"],
            partition["join_Npix"],
        )
        table = pq.read_table(input_dir)
        rows_written = len(table)

        if rows_written != partition["num_rows"]:
            raise ValueError(
                "Unexpected number of objects ",
                f" Expected {partition['num_rows']}, wrote {rows_written}",
            )

        table.to_pandas().set_index("primary_hipscat_index").sort_index().to_parquet(
            output_file
        )

    return data_frame["num_rows"].sum()
