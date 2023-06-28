Catalog Import Arguments
===============================================================================

This page discusses a few topics around setting up a catalog pipeline.

For a full list of the available arguments, see the API documentation for 
:py:class:`hipscat_import.catalog.arguments.ImportArguments`

Reading input files
-------------------------------------------------------------------------------

Catalog import reads through a list of files and converts them into a hipscatted catalog.

Which files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few ways to specify the files to read:

* ``input_path`` + ``input_format``: 
    will search for files ending with the format string in the indicated directory.
* ``input_file_list``: 
    a list of fully-specified paths you want to read.

    * this strategy can be useful to first run the import on a single input
      file and validate the input, then run again on the full input set, or 
      to debug a single input file with odd behavior. 
    * if you have a mix of files in your target directory, you can use a glob
      statement like the following to gather input files:

.. code-block:: python

    in_file_paths = glob.glob("/data/object_and_source/object**.csv")
    in_file_paths.sort()

How to read them?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify an instance of ``InputReader`` for the ``file_reader`` parameter.

see the API documentation for 
:py:class:`hipscat_import.catalog.file_readers.InputReader`

We use the ``InputReader`` class to read files in chunks and pass the chunks
along to the map/reduce stages. We've provided reference implementations for 
reading CSV, FITS, and Parquet input files, but you can subclass the reader 
type to suit whatever input files you've got.

.. code-block:: python

    class StarrReader(InputReader):
        """Class for fictional Starr file format."""
        def __init__(self, chunksize=500_000, **kwargs):
            self.chunksize = chunksize
            self.kwargs = kwargs

        def read(self, input_file):
            starr_file = starr_io.read_table(input_file, **self.kwargs)
            for smaller_table in starr_file.to_batches(max_chunksize=self.chunksize):
                smaller_table = filter_nonsense(smaller_table)
                yield smaller_table.to_pandas()

        def provenance_info(self) -> dict:
            provenance_info = {
                "input_reader_type": "StarrReader",
                "chunksize": self.chunksize,
            }
            return provenance_info

    ...

    args = ImportArguments(
        ...
        ## Locates files like "/directory/to/files/**starr"
        input_path="/directory/to/files/",
        input_format="starr",
        ## NB - you need the parens here!
        file_reader=StarrReader(),

    )

Which fields?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the ``ra_column`` and ``dec_column`` for the dataset.

There are two fields that we require in order to make a valid hipscatted
catalog, the right ascension and declination. At this time, this is the only 
supported system for celestial coordinates.


Healpix order and thresholds
-------------------------------------------------------------------------------

Details for ``pixel_threshold``, ``highest_healpix_order``, and
``constant_healpix_order`` arguments

When creating a new catalog through the hipscat-import process, we try to 
create partitions with approximately the same number of rows per partition. 
This isn't perfect, because the sky is uneven, but we still try to create 
smaller-area pixels in more dense areas, and larger-area pixels in less dense 
areas. 

We use the argument ``pixel_threshold`` and will split a partition into 
smaller healpix pixels until the number of rows is smaller than ``pixel_threshold``.
We will only split by healpix pixels up to the ``highest_healpix_order``. If we
would need to split further, we'll throw an error at the "Binning" stage, and you 
should adjust your parameters.

For more discussion of the ``pixel_threshold`` argument and a strategy for setting
this parameter, see notebook :doc:`/notebooks/estimate_pixel_threshold`

Alternatively, you can use the ``constant_healpix_order`` argument. This will 
**ignore** both of the ``pixel_threshold`` and ``highest_healpix_order`` arguments
and the catalog will be partitioned by healpix pixels at the
``constant_healpix_order``. This can be useful for very sparse datasets.