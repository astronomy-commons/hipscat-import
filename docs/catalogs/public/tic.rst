TIC
===============================================================================

Getting the data
-------------------------------------------------------------------------------

Tess Input Catalog. See the data at NASA.

https://tess.mit.edu/science/tess-input-catalogue/

Challenges with this data set
-------------------------------------------------------------------------------

- The individual files are large, and so we want to use a chunked CSV reader.
- The rows are wide, so the chunked reader cannot read too many rows at once.
- The CSV files don't have a header, so we need to provide the column names and
  type hints to the reader.
- The numeric fields may be null, which is not directly supported by the 
  ``int64`` type in pandas, so we must use the nullable ``Int64`` type.

You can download our reference files, if you find that helpful:

- :download:`tic_types</static/tic_types.csv>` CSV file with names and types
- :download:`tic_schema</static/tic_schema.parquet>` column-level parquet metadata

Example import
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hats_import.pipeline as runner
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.catalog.file_readers import CsvReader

    type_frame = pd.read_csv("tic_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))
    
    args = ImportArguments(
        output_artifact_name="tic_1",
        input_path="/path/to/tic/",
        file_reader=CsvReader(
            header=None,
            column_names=type_frame["name"].values.tolist(),
            type_map=type_map,
            chunksize=250_000,
        ).read,
        ra_column="ra",
        dec_column="dec",
        sort_columns="ID",
        output_path="/path/to/catalogs/",
        use_schema_file="tic_schema.parquet",
    )
    runner.run(args)