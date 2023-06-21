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

You can download the :download:`neowise_types</static/neowise_types.csv>` CSV file we used.

Example import
-------------------------------------------------------------------------------

.. code-block:: python

    import pandas as pd

    import hipscat_import.pipeline as runner
    from hipscat_import.catalog.arguments import ImportArguments
    from hipscat_import.catalog.file_readers import CsvReader

    # Load the column names and types from a side file.
    type_frame = pd.read_csv("neowise_types.csv")
    type_map = dict(zip(type_frame["name"], type_frame["type"]))

    args = ImportArguments(
        output_catalog_name="neowise_1",
        input_path="/path/to/neowiser_year8/",
        input_format="csv.bz2",
        file_reader=CsvReader(
            header=None,
            separator="|",
            column_names=type_frame["name"].values.tolist(),
            type_map=type_map,
            chunksize=250_000,
        ).read,
        ra_column="RA",
        dec_column="DEC",
        id_column="SOURCE_ID",
        output_path="/path/to/catalogs/",
    )
    runner.run(args)
