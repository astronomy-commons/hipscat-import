"""Top-level testing configuration."""

import os

import pytest
import ray
from dask.distributed import Client
from ray.util.dask import disable_dask_on_ray, enable_dask_on_ray

# pylint: disable=missing-function-docstring, redefined-outer-name


@pytest.fixture(scope="session", name="use_ray")
def use_ray(request):
    return request.config.getoption("--use_ray")


@pytest.fixture(scope="session", name="dask_client")
def dask_client(use_ray):
    """Create a single client for use by all unit test cases."""
    if use_ray:
        ray.init(num_cpus=1)
        enable_dask_on_ray()
        ## Default values that are not set in a pytest environment.
        os.environ["RAY_memory_monitor_refresh_ms"] = "250"
        os.environ["RAY_memory_usage_threshold"] = ".95"

        client = Client()
        yield client
        client.close()

        disable_dask_on_ray()
    else:
        client = Client()
        yield client
        client.close()


def pytest_addoption(parser):
    """Add command line option to test dask unit tests on ray.

    This must live in /tests/conftest.py (not /tests/hipscat-import/conftest.py)"""
    parser.addoption(
        "--use_ray",
        action="store_true",
        default=False,
        help="use dask-on-ray for dask tests",
    )
