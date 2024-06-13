import logging

import pytest

import hipscat_import.verification.run_verification as runner
from hipscat_import.verification.arguments import VerificationArguments


def test_bad_args():
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(None)

    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(args)


def test_no_implementation(tmp_path, small_sky_object_catalog, caplog):
    """Womp womp. Test that we don't have a verification pipeline implemented"""
    caplog.set_level(logging.INFO, logger="hipscat_verification")
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )

    runner.run(args)

    print(caplog.text)
