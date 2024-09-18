NEOWISE
===============================================================================

Getting the data
-------------------------------------------------------------------------------

See docs at IRSA.

https://irsa.ipac.caltech.edu/data/download/neowiser_year8/

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

- :download:`neowise_types</static/neowise_types.csv>` CSV file with names and types
- :download:`neowise_schema</static/neowise_schema.parquet>` column-level parquet metadata

Example import
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hats_import.pipeline as runner
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.catalog.file_readers import CsvReader

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("neowise_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))

    args = ImportArguments(
        output_artifact_name="neowise_1",
        input_path="/path/to/neowiser_year8/",
        file_reader=CsvReader(
            header=None,
            sep="|",
            column_names=type_frame["name"].values.tolist(),
            type_map=type_map,
            chunksize=250_000,
        ).read,
        ra_column="RA",
        dec_column="DEC",
        pixel_threshold=2_000_000,
        highest_healpix_order=9,
        use_schema_file="neowise_schema.parquet",
        sort_columns="SOURCE_ID",
        output_path="/path/to/catalogs/",
    )
    runner.run(args)


Our performance
-------------------------------------------------------------------------------

Running on dedicated hardware, with 10 workers, this import took around
a week.

- Mapping stage: around 58 hours
- Splitting stage: around 72 hours
- Reducing stage: around 14 hours
