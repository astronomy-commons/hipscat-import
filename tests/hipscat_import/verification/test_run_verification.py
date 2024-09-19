from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hipscat_import.verification.run_verification as runner
from hipscat_import.verification.arguments import VerificationArguments


def test_verifier_with_valid_catalog(tmp_path, small_sky_object_catalog):
    """Verifier tests should pass with valid catalogs"""
    args = VerificationArguments(
        input_catalog_path=small_sky_object_catalog,
        output_path=tmp_path,
        use_schema_file=small_sky_object_catalog / "_metadata",
        expected_total_rows=131,
        # next arg exists but is currently unused. we just get the distributions of all fields.
        # field_distribution_cols=[],
    )
    _test_verifier(args, assert_passed=True)


def test_verifier_with_invalid_catalog(tmp_path, invalid_catalog):
    """Verifier tests should fail with invalid catalogs"""
    args = VerificationArguments(
        input_catalog_path=invalid_catalog,
        output_path=tmp_path,
        use_schema_file=invalid_catalog / "_metadata",
        expected_total_rows=131,
        # next arg exists but is currently unused. we just get the distributions of all fields.
        # field_distribution_cols=[],
    )
    _test_verifier(args, assert_passed=False)


def _test_verifier(args, assert_passed=True):
    verifier = runner.run(args, write_mode="w")
    _test_catalog_results(results=verifier.results_df, assert_passed=assert_passed)
    _test_schema_results(results=verifier.results_df, assert_passed=assert_passed)
    _test_row_count_results(results=verifier.results_df, assert_passed=assert_passed)
    _test_verifier_reports_exist(verifier)


def _test_catalog_results(results, assert_passed=True):
    """hipscat is_valid_catalog should have passed."""
    catalog_res = results.loc[results.test == "is valid catalog"]
    _passed = catalog_res.passed.squeeze()
    assert _passed if assert_passed else ~_passed


def _test_schema_results(results, assert_passed=True):
    """Schema checks should have passed."""
    schema_res = results.loc[results.test == "schema"]

    _passed = schema_res.loc[schema_res.targets == "_common_metadata vs input"].passed.squeeze()
    assert _passed if assert_passed else ~_passed

    _passed = schema_res.loc[schema_res.targets == "file footers vs input"].passed.squeeze()
    assert _passed if assert_passed else ~_passed

    _passed = schema_res.loc[schema_res.targets == "_metadata vs input"].passed.squeeze()
    assert _passed if assert_passed else ~_passed


def _test_row_count_results(results, assert_passed=True):
    """Row-count checks should have passed."""
    nrows_res = results.loc[results.test == "num rows"]

    _passed = nrows_res.loc[nrows_res.targets == "_metadata vs file footers"].passed.squeeze()
    assert _passed if assert_passed else ~_passed

    _passed = nrows_res.loc[nrows_res.targets == "user total vs file footers"].passed.squeeze()
    assert _passed if assert_passed else ~_passed


def _test_verifier_reports_exist(verifier):
    """Verifier should have written two reports to file."""
    # [FIXME] handle paths better.

    # verifier.record_results() writes this file
    results = pd.read_csv(Path(verifier.args.output_path) / "verifier_results.csv")
    assert results.equals(verifier.results_df)

    # verifier.record_distributions() writes this file
    distributions = pd.read_csv(
        Path(verifier.args.output_path) / "field_distributions.csv", index_col="field"
    )
    # values are floats, so use np.allclose
    assert np.allclose(distributions.minimum, verifier.distributions_df.minimum, equal_nan=True)
    assert np.allclose(distributions.maximum, verifier.distributions_df.maximum, equal_nan=True)


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
