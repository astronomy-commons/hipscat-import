AllWISE
===============================================================================

Getting the data
-------------------------------------------------------------------------------

See docs at IRSA.

https://irsa.ipac.caltech.edu/data/download/wise-allwise/

Challenges with this data set
-------------------------------------------------------------------------------

- The individual files are large, and so we want to use a chunked CSV reader.
- The rows are wide, so the chunked reader cannot read too many rows at once.
- The CSV files don't have a header, so we need to provide the column names and
  type hints to the reader.
- The numeric fields may be null, which is not directly supported by the 
  ``int64`` type in pandas, so we must use the nullable ``Int64`` type.
- Some fields are sparsely populated, and this can create type conversion issues.
  We use a schema parquet file to address these issues.

You can download our reference files, if you find that helpful:

- :download:`allwise_types</static/allwise_types.csv>` CSV file with names and types
- :download:`allwise_schema</static/allwise_schema.parquet>` column-level parquet metadata

Example import
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hats_import.pipeline as runner
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.catalog.file_readers import CsvReader

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("allwise_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))

    args = ImportArguments(
        output_artifact_name="allwise",
        input_path="/path/to/allwise/",
        file_reader=CsvReader(
            chunksize=250_000,
            header=None,
            sep="|",
            column_names=type_frame["name"].values.tolist(),
            type_map=type_map,
        ),
        use_schema_file="allwise_schema.parquet",
        ra_column="ra",
        dec_column="dec",
        sort_columns="source_id",
        pixel_threshold=1_000_000,
        highest_healpix_order=7,
        output_path="/path/to/catalogs/",
    )
    runner.pipeline(args)
