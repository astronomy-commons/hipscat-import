from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hipscat_import.verification.run_verification as runner
from tests.hipscat_import.verification.fixture import VerifierFixture


def test_bad_args():
    """Runner should fail with empty or mis-typed arguments"""
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(None)

    args = {"output_artifact_name": "bad_arg_type"}
    with pytest.raises(TypeError, match="VerificationArguments"):
        runner.run(args)


def test_basic_run(verifier_for_runner):
    """Verification runner should execute all tests and write reports to file.
    Tests should pass with valid catalogs and fail with malformed catalogs."""
    args = verifier_for_runner.verifier.args
    # start fresh. delete any existing output files.
    filenames = [args.output_report_filename, args.output_distributions_filename]
    [(args.output_path / filename).unlink(missing_ok=True) for filename in filenames]

    # run the tests
    verifier = runner.run(args)

    # Show that the verification passed or failed as expected
    tests_passed = verifier.results_df.passed.all()
    assert tests_passed == verifier_for_runner.assert_passed, "runner tests"

    # Show that the output files were or were not written as expected
    all_output_written = True
    try:
        _check_file_output(verifier)
    except AssertionError:
        all_output_written = False
    assert all_output_written == verifier_for_runner.assert_passed, "runner output"


def _check_file_output(verifier: runner.Verifier) -> None:
    """Verifier should have written two reports to file."""
    # verifier.record_results() writes this file
    freport = verifier.args.output_path / verifier.args.output_report_filename
    assert freport.is_file(), f"File not found {freport}"
    results = pd.read_csv(freport)
    # the affected_files lists cause problems. just exclude them
    cols = [c for c in results.columns if not c == "affected_files"]
    assert results[cols].equals(verifier.results_df[cols]), "Mismatched results"

    # verifier.test_rowgroup_stats() writes this file
    fdistributions = verifier.args.output_path / verifier.args.output_distributions_filename
    assert fdistributions.is_file(), f"File not found {fdistributions}"
    distributions = pd.read_csv(fdistributions, index_col="field")
    # values are floats, so use np.allclose
    min_passed = np.allclose(distributions.minimum, verifier.distributions_df.minimum, equal_nan=True)
    max_passed = np.allclose(distributions.maximum, verifier.distributions_df.maximum, equal_nan=True)
    assert min_passed and max_passed, "Mismatched distributions"


def test_test_file_sets(verifier_for_file_sets):
    """Files on disk should match files in _metadata for catalogs that are not malformed."""
    # run the test
    verifier = verifier_for_file_sets.verifier
    verifier.results = []  # ensure a fresh start
    verifier.test_file_sets()

    # check the result
    result = verifier.results_df.squeeze()
    _check_one_result(result, verifier_for_file_sets.assert_passed, "file_sets")


def test_test_is_valid_catalog(verifier_for_is_valid_catalog):
    """hipscat's is_valid_catalog should pass for valid catalogs, else fail."""
    # run the test
    verifier = verifier_for_is_valid_catalog.verifier
    verifier.results = []  # ensure a fresh start
    verifier.test_is_valid_catalog()

    # check the result
    result = verifier.results_df.squeeze()
    _check_one_result(result, verifier_for_is_valid_catalog.assert_passed, "is_valid_catalog")


def test_test_num_rows(verifier_for_num_rows):
    """Row count tests should pass for catalogs that are not malformed."""
    # run the test
    verifier = verifier_for_num_rows.verifier
    verifier.results = []  # ensure a fresh start
    verifier.test_num_rows()

    # check the results
    targets = verifier_for_num_rows.test_targets["num_rows"]
    _check_results(verifier_for_num_rows, targets)


def test_test_rowgroup_stats(verifier_for_rowgroup_stats):
    """Row group statistics should be present in _metadata for files that are not malformed."""
    # run the test
    verifier = verifier_for_rowgroup_stats.verifier
    verifier.results = []  # ensure a fresh start
    verifier.test_rowgroup_stats()

    # check the result
    result = verifier.results_df.squeeze()
    _check_one_result(result, verifier_for_rowgroup_stats.assert_passed, test_name="rowgroup_stats")


def test_test_schemas(verifier_for_schemas):
    """Schemas should contain correct columns, dtypes, and metadata for catalogs that are not malformed."""
    # run the tests
    verifier = verifier_for_schemas.verifier
    verifier.results = []  # ensure a fresh start
    verifier.test_schemas()

    # Two tests were run ('schema' and 'schema metadata') with several targets per test.
    test_targets = verifier_for_schemas.test_targets["schema"]  # dict maps test -> targets
    assert_passed = verifier_for_schemas.unpack_assert_passed(  # dict maps test -> assertion
        verifier_for_schemas.assert_passed, targets=test_targets.keys()
    )

    # Check results for each test separately.
    for test, targets in test_targets.items():
        results = verifier.results_df.loc[verifier.results_df.test == test]
        _check_results(verifier_for_schemas, targets, results=results, assert_passed=assert_passed[test])


def _check_results(
    verifier_fixture: VerifierFixture,
    targets: list,
    *,
    results: pd.DataFrame | None = None,
    assert_passed: bool | dict | None = None,
) -> None:
    """Check the results of verification tests for the given targets.

    Parameters
    ----------
        verifier_fixture : VerifierFixture
            The fixture containing the verifier and its results.
        targets : list
            The list of test targets to check. There should be one result per target.
        results : pd.DataFrame or None
            The test results to check. If None, verifier_fixture.verifier.results_df will be used.
        assert_passed : bool, dict, or None
            Whether the test should have passed for each target. If None,
            verifier_fixture.assert_passed is used.

    Raises
    ------
        AssertionError: If any results are unexpected.
    """
    results = verifier_fixture.verifier.results_df if results is None else results
    assert_passed = verifier_fixture.assert_passed if assert_passed is None else assert_passed

    # dict with one entry per target
    _assert_passed = verifier_fixture.unpack_assert_passed(assert_passed, targets=targets)
    for target, assertion in _assert_passed.items():
        # Expecting one result per target so squeeze to a series
        result = results.loc[results.target.str.startswith(target)].squeeze()
        _check_one_result(result, assertion, test_name=target)


def _check_one_result(result: pd.Series, assertion: bool | dict | None, test_name: str) -> None:
    """Check the result of a single verification test.

    Parameters
    ----------
        result : pd.Series
            Test result reported by the verifier.
        assertion : bool, or dict, or None
            The expected outcome of the test. None indicates that the test should have been skipped.
            A boolean indicates a simple pass/fail. A dict indicates expected failure and the
            list of file suffixes expected in the result's affected_files field.
        test_name : str
            The name of the test being verified.

    Raises
    ------
        AssertionError: If the result does not match the assertion.
    """
    if assertion is None:
        # This test should have been skipped
        msg = f"Unexpected result for: {test_name}. There is probably a bug in the code."
        assert len(result.passed) == 0, msg
        return

    assert_passed, bad_suffixes = VerifierFixture.unpack_assert_passed(assertion)

    # Show that the target passed or failed the test as expected
    assert result.passed if assert_passed else not result.passed, test_name

    # Show that all files that should have failed the test actually did, and no more.
    # We're only trying to match file suffixes so strip the rest of the file path out of results.
    found_suffixes = ["".join(Path(file).suffixes) for file in result.affected_files]
    assert set(bad_suffixes) == set(found_suffixes), test_name + " affected_files"
