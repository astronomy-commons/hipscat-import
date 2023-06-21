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

You can download the :download:`allwise_types</static/allwise_types.csv>` CSV file we used,
and the associated schema file :download:`allwise_schema</static/allwise_schema.parquet>`
with column-level parquet metadata.

Example import
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hipscat_import.pipeline as runner
    from hipscat_import.catalog.arguments import ImportArguments
    from hipscat_import.catalog.file_readers import CsvReader

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("allwise_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))

    args = ImportArguments(
        output_catalog_name="allwise",
        input_path="/path/to/allwise/",
        input_format="csv.bz2",
        file_reader=CsvReader(
            header=None,
            separator="|",
            column_names=type_frame["name"].values.tolist(),
            type_map=type_map,
            chunksize=250_000,
        ),
        use_schema_file="allwise_schema.parquet",
        ra_column="ra",
        dec_column="dec",
        id_column="source_id",
        pixel_threshold=1_000_000,
        highest_healpix_order=7,
        output_path="/path/to/catalogs/",
    )
    runner.pipeline(args)
