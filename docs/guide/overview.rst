Getting Started
===============================================================================

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

        $ conda config --add channels conda-forge
        $ conda install healpy

Setting up a pipeline
-------------------------------------------------------------------------------

For each type of dataset the hipscat-import tool can generate, there is an argument
container class that you will need to instantiate and populate with relevant arguments.

See dataset-specific notes on arguments:

* :doc:`catalog_arguments` (most common)
* :doc:`margin_cache`
* :doc:`association`
* :doc:`index_table`

Once you have created your arguments object, you pass it into the pipeline control,
and then wait:

.. code-block:: python

    import hipscat_import.control as runner

    args = ...
    runner.run(args)
