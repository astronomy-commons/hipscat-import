Using dask-on-ray
===============================================================================

What is it?
-------------------------------------------------------------------------------

See more on Ray's site:

https://docs.ray.io/en/latest/ray-more-libs/dask-on-ray.html

How to use in hats-import pipelines
-------------------------------------------------------------------------------

Install ray

.. code-block:: python

    pip install ray

Create your client within a ray initialization context and enable dask_on_ray.

You should also disable ray when you're done, just to clean things up.

.. code-block:: python

    import ray
    from dask.distributed import Client
    from ray.util.dask import disable_dask_on_ray, enable_dask_on_ray

    from hats_import.pipeline import pipeline_with_client

    with ray.init(
        num_cpus=args.dask_n_workers,
        _temp_dir=args.dask_tmp,
    ):
        enable_dask_on_ray()

        with Client(
            local_directory=args.dask_tmp,
            n_workers=args.dask_n_workers
        ) as client:
            pipeline_with_client(args, client)

        disable_dask_on_ray()

Your pipeline should execute as though it were using ray for task workers.