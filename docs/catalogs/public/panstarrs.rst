PanStarrs
===============================================================================

Getting the data
-------------------------------------------------------------------------------

We had a special line to the folks at NASA to get a hold of the full object 
catalog, and the detections table. This is provided for reference.

Challenges with this data set
-------------------------------------------------------------------------------

- The rows are wide, so the chunked reader cannot read too many rows at once.
- The CSV files don't have a header, so we need to provide the column names and
  type hints to the reader.
- The tables are very wide. We only used a subset of columns in each table for
  our initial science use cases.

You can download the CSV files we used that contain python type information:

- :download:`ps1_otmo_types</static/ps1_otmo_types.csv>`
- :download:`ps1_detections_types</static/ps1_detections_types.csv>`
- :download:`ps1_stack_object_types</static/ps1_stack_object_types.csv>`
- :download:`ps1_forced_mean_object_types</static/ps1_forced_mean_object_types.csv>`

Example import of objects (otmo)
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hats_import.pipeline as runner
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.catalog.file_readers import CsvReader

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("ps1_otmo_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))
    type_names = type_frame["name"].values.tolist()

    in_file_paths = glob.glob("/path/to/otmo/OTMO_**.csv")
    in_file_paths.sort()
    args = ImportArguments(
        output_artifact_name="ps1_otmo",
        input_file_list=in_file_paths,
        file_reader=CsvReader(
            header=None,
            index_col=False,
            column_names=type_names,
            type_map=type_map,
            chunksize=250_000,
            usecols=use_columns,
        ),
        ra_column="raMean",
        dec_column="decMean",
        sort_columns="objID",
    )
    runner.pipeline(args)


Example import of detections
-------------------------------------------------------------------------------

.. code-block:: python

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("ps1_detections_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))
    type_names = type_frame["name"].values.tolist()

    in_file_paths = glob.glob("/path/to/detection/detection**.csv")
    in_file_paths.sort()
    args = ImportArguments(
        output_artifact_name="ps1_detection",
        input_file_list=in_file_paths,
        file_reader=CsvReader(
            header=None,
            index_col=False,
            column_names=type_names,
            type_map=type_map,
            chunksize=250_000,
            usecols=use_columns,
        ),
        ra_column="ra",
        dec_column="dec",
        sort_columns="objID",
    )
    runner.pipeline(args)
