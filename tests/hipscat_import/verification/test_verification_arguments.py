"""Tests of argument validation"""

import pytest
from hipscat.catalog import Catalog

from hipscat_import.verification.arguments import VerificationArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        VerificationArguments()


def test_empty_required(tmp_path):
    """*Most* required arguments are provided."""
    ## Input path is missing
    with pytest.raises(ValueError, match="input_catalog_path"):
        VerificationArguments(
            output_path=tmp_path,
            output_artifact_name="small_sky_object_verification_report",
        )


def test_invalid_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )

    ## Input path is invalid catalog
    with pytest.raises(ValueError, match="input_catalog_path not a valid catalog"):
        VerificationArguments(
            input_catalog_path="path",
            output_path=f"{tmp_path}/path",
            output_artifact_name="small_sky_object_verification_report",
        )


def test_good_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )
    assert args.input_catalog_path == small_sky_object_catalog
    assert str(args.output_path) == tmp_path_str
    assert str(args.tmp_path).startswith(tmp_path_str)


def test_catalog_object(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, and paths are found."""
    small_sky_catalog_object = Catalog.read_from_hipscat(catalog_path=small_sky_object_catalog)
    tmp_path_str = str(tmp_path)
    args = VerificationArguments(
        input_catalog=small_sky_catalog_object,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )
    assert args.input_catalog_path == small_sky_object_catalog
    assert str(args.output_path) == tmp_path_str
    assert str(args.tmp_path).startswith(tmp_path_str)


@pytest.mark.timeout(5)
def test_provenance_info(small_sky_object_catalog, tmp_path):
    """Verify that provenance info includes verification-specific fields.
    NB: This is currently the last test in alpha-order, and may require additional
    time to teardown fixtures."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "input_catalog_path" in runtime_args
