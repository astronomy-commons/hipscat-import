HiPSCat Import
========================================================================================

Utility for ingesting large survey data into HiPSCat structure.

Installation
-------------------------------------------------------------------------------

We recommend installing in a virtual environment, like venv or conda. You may
need to install or upgrade versions of dependencies to work with hipscat-import.

.. code-block:: console

    pip install hipscat-import

.. tip::
    Installing on Mac

    ``healpy`` is a very necessary dependency for hipscat libraries at this time, but
    native prebuilt binaries for healpy on Apple Silicon Macs 
    `do not yet exist <https://healpy.readthedocs.io/en/latest/install.html#binary-installation-with-pip-recommended-for-most-other-python-users>`_, 
    so it's recommended to install via conda before proceeding to hipscat-import.

    .. code-block:: console

        >> conda config --append channels conda-forge
        >> conda install healpy

Setting up a pipeline
-------------------------------------------------------------------------------

For each type of dataset the hipscat-import tool can generate, there is an argument
container class that you will need to instantiate and populate with relevant arguments.

See dataset-specific notes on arguments:

* :doc:`catalogs/arguments` (most common)
* :doc:`guide/margin_cache`
* :doc:`guide/index_table`

Once you have created your arguments object, you pass it into the pipeline control,
and then wait. Running within a main guard will potentially avoid some python
threading issues with dask:

.. code-block:: python

    from dask.distributed import Client
    from hipscat_import.pipeline import pipeline_with_client

    def main():
        args = ...
        with Client(
            n_workers=10,
            threads_per_worker=1,
            ... 
        ) as client:
            pipeline_with_client(args, client)

    if __name__ == '__main__':
        main()

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Catalogs

   catalogs/arguments
   catalogs/temp_files
   catalogs/public/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Other Datasets

   guide/margin_cache
   guide/index_table   
   Notebooks <notebooks>
   guide/dask_on_ray

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developers

   guide/contributing
   API Reference <autoapi/index>
   guide/contact
