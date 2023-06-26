HiPSCat Import
========================================================================================

Utility for ingesting large survey data into HiPSCat structure.

Installation
-------------------------------------------------------------------------------

.. code-block:: bash

    pip install hipscat-import

.. tip::
    Installing on Mac

    ``healpy`` is a very necessary dependency for hipscat libraries at this time, but
    native prebuilt binaries for healpy on Apple Silicon Macs 
    `do not yet exist <https://healpy.readthedocs.io/en/latest/install.html#binary-installation-with-pip-recommended-for-most-other-python-users>`_, 
    so it's recommended to install via conda before proceeding to hipscat-import.

    .. code-block:: bash

        $ conda config --append channels conda-forge
        $ conda install healpy

Setting up a pipeline
-------------------------------------------------------------------------------

For each type of dataset the hipscat-import tool can generate, there is an argument
container class that you will need to instantiate and populate with relevant arguments.

See dataset-specific notes on arguments:

* :doc:`catalogs/arguments` (most common)
* :doc:`guide/margin_cache`
* :doc:`guide/association`
* :doc:`guide/index_table`

Once you have created your arguments object, you pass it into the pipeline control,
and then wait:

.. code-block:: python

    import hipscat_import.pipeline as runner

    args = ...
    runner.pipeline(args)


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Catalogs

   catalogs/arguments
   catalogs/resume
   catalogs/debug
   catalogs/advanced
   catalogs/public/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Other Datasets

   guide/margin_cache
   guide/association
   guide/index_table   
   Notebooks <notebooks>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developers

   guide/contributing
   API Reference <autoapi/index>
   guide/contact
