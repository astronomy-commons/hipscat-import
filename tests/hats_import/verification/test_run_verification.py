import pytest

import hats_import.verification.run_verification as runner
from hats_import.verification.arguments import VerificationArguments


def test_bad_args():
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(None)

    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(args)


def test_no_implementation(tmp_path, small_sky_object_catalog):
    """Womp womp. Test that we don't have a verification pipeline implemented"""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        output_artifact_name="small_sky_object_verification_report",
    )
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        runner.run(args)
