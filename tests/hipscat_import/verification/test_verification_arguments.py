"""Tests of argument validation"""


import pytest

from hipscat_import.verification.arguments import VerificationArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        VerificationArguments()


def test_empty_required(tmp_path, small_sky_object_catalog):
    """*Most* required arguments are provided."""
    ## Input path is missing
    with pytest.raises(ValueError, match="input_catalog_path"):
        VerificationArguments(
            output_path=tmp_path,
            output_catalog_name="small_sky_object_verification_report",
        )


def test_invalid_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, but paths aren't found."""
    ## Prove that it works with required args
    VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_catalog_name="small_sky_object_verification_report",
    )

    ## Bad input path
    with pytest.raises(FileNotFoundError):
        VerificationArguments(
            input_catalog_path="path",
            output_path="path",
            output_catalog_name="small_sky_object_verification_report",
        )


def test_good_paths(tmp_path, small_sky_object_catalog):
    """Required arguments are provided, and paths are found."""
    tmp_path_str = str(tmp_path)
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_catalog_name="small_sky_object_verification_report",
    )
    assert args.input_catalog_path == small_sky_object_catalog
    assert str(args.output_path) == tmp_path_str
    assert str(args.tmp_path).startswith(tmp_path_str)


def test_provenance_info(small_sky_object_catalog, tmp_path):
    """Verify that provenance info includes verification-specific fields."""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_catalog_name="small_sky_object_verification_report",
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "input_catalog_path" in runtime_args
