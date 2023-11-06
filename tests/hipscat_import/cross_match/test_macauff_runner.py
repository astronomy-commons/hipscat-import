import pytest

import hipscat_import.cross_match.run_macauff_import as runner
from hipscat_import.cross_match.macauff_arguments import MacauffArguments

# pylint: disable=too-many-instance-attributes
# pylint: disable=duplicate-code


@pytest.mark.dask
def test_bad_args(dask_client):
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="MacauffArguments"):
        runner.run(None, dask_client)

    args = {"output_catalog_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="MacauffArguments"):
        runner.run(args, dask_client)


@pytest.mark.dask
def test_no_implementation(
    small_sky_object_catalog,
    small_sky_source_catalog,
    small_sky_dir,
    formats_yaml,
    tmp_path,
    dask_client,
):
    """Test that we can create a MacauffArguments instance with two valid catalogs."""
    args = MacauffArguments(
        output_path=tmp_path,
        output_catalog_name="object_to_source",
        tmp_dir=tmp_path,
        left_catalog_dir=small_sky_object_catalog,
        left_ra_column="ra",
        left_dec_column="dec",
        left_id_column="id",
        right_catalog_dir=small_sky_source_catalog,
        right_ra_column="source_ra",
        right_dec_column="source_dec",
        right_id_column="source_id",
        input_path=small_sky_dir,
        input_format="csv",
        metadata_file_path=formats_yaml,
    )

    with pytest.raises(NotImplementedError, match="not implemented yet."):
        runner.run(args, dask_client)
