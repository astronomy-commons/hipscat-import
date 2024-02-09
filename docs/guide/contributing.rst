Contributing to hipscat-import
===============================================================================

Find (or make) a new GitHub issue
-------------------------------------------------------------------------------

Add yourself as the assignee on an existing issue so that we know who's working 
on what. (If you're not actively working on an issue, unassign yourself).

If there isn't an issue for the work you want to do, please create one and include
a description.

You can reach the team with bug reports, feature requests, and general inquiries
by creating a new GitHub issue.

.. tip::
   Want to help?

   Do you want to help out, but you're not sure how? :doc:`/guide/contact`

Create a branch
-------------------------------------------------------------------------------

It is preferable that you create a new branch with a name like 
``issue/##/<short-description>``. GitHub makes it pretty easy to associate 
branches and tickets, but it's nice when it's in the name.

Setting up a development environment
-------------------------------------------------------------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: console

   >> conda create env -n <env_name> python=3.10
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: console

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.


.. tip::
    Installing on Mac

    ``healpy`` is a very necessary dependency for hipscat libraries at this time, but
    native prebuilt binaries for healpy on Apple Silicon Macs 
    `do not yet exist <https://healpy.readthedocs.io/en/latest/install.html#binary-installation-with-pip-recommended-for-most-other-python-users>`_, 
    so it's recommended to install via conda before proceeding to hipscat-import.

    .. code-block:: console

        >> conda config --add channels conda-forge
        >> conda install healpy
        >> git clone https://github.com/astronomy-commons/hipscat-import
        >> cd hipscat-import
        >> pip install -e .
        
    When installing dev dependencies, make sure to include the single quotes.

    .. code-block:: console
        
        >> pip install -e '.[dev]'

Testing
-------------------------------------------------------------------------------

We use ``pytest`` as our preferred unit test runner engine, in keeping with
LSST DM style guide. We make heavy use of 
`pytest fixtures <https://docs.pytest.org/en/7.1.x/explanation/fixtures.html#about-fixtures>`_, 
which set up various resources used for unit testing, or provide consistent 
paths. These are defined in ``conftest.py`` files. They're powerful and flexible 
(and fun in their own way), and we encourage contributors to familiarize themselves.

Please add or update unit tests for all changes made to the codebase. You can run
unit tests locally simply with:

.. code-block:: console

    >> pytest

If you're making changes to the sphinx documentation (anything under ``docs``),
you can build the documentation locally with a command like:

.. code-block:: console

    >> cd docs
    >> make html

Create your PR
-------------------------------------------------------------------------------

Please use PR best practices, and get someone to review your code.

The LINCC Frameworks guidelines and philosophy on code reviews can be found on 
`our wiki <https://github.com/lincc-frameworks/docs/wiki/Design-and-Code-Review-Policy>`_.

We have a suite of continuous integration tests that run on PR creation. Please
follow the recommendations of the linter.

Merge your PR
-------------------------------------------------------------------------------

The author of the PR is welcome to merge their own PR into the repository.

Optional - Release a new version
-------------------------------------------------------------------------------

Once your PR is merged you can create a new release to make your changes available. 
GitHub's `instructions <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>`_ for doing so are here. 
Use your best judgement when incrementing the version. i.e. is this a major, minor, or patch fix.
