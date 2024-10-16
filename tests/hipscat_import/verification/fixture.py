"""Run pass/fail tests and generate verification report of existing hipscat table."""

from pathlib import Path

import attrs
import yaml

from hipscat_import.verification.arguments import VerificationArguments
from hipscat_import.verification.run_verification import Verifier


@attrs.define
class VerifierFixture:
    """Class for pytest fixtures for verification tests. Instantiate using the 'from_param' method."""

    test_targets: dict[str, list | dict] = attrs.field(validator=attrs.validators.instance_of(dict))
    """Dictionary mapping test names to targets."""
    verifier: Verifier = attrs.field(validator=attrs.validators.instance_of(Verifier))
    """Verifier instance that the fixture will use to run verification tests."""
    assert_passed: bool | dict = attrs.field(validator=attrs.validators.instance_of((bool, dict)))
    """Expected result(s) of the test(s) this verifier will run."""

    @classmethod
    def from_param(
        cls, fixture_param: str, malformed_catalog_dirs: dict[str, Path], tmp_path: Path
    ) -> "VerifierFixture":
        """Create a VerifierFixture from the given fixture parameter.

        Fixture definitions, including the expected test outcomes, are defined in fixture_defs.yaml.

        Parameters
        ----------
            fixture_param : str
                The fixture parameter key to look up fixture definitions.
            malformed_catalog_dirs : dict[str, Path]
                A mapping of malformed test dataset names to their directories.
            tmp_path : Path
                A temporary path for output.

        Returns:
            VerifierFixture: An instance of VerifierFixture configured with the specified parameters.
        """
        with open(Path(__file__).parent / "fixture_defs.yaml", "r") as fin:
            fixture_defs = yaml.safe_load(fin)
        fixture_def = fixture_defs[fixture_param]

        truth_schema = fixture_def.get("truth_schema")
        if truth_schema is not None:
            truth_schema = malformed_catalog_dirs[truth_schema.split("/")[0]] / truth_schema.split("/")[1]
        args = VerificationArguments(
            input_catalog_path=malformed_catalog_dirs[fixture_def["input_dir"]],
            output_path=tmp_path,
            truth_schema=truth_schema,
            truth_total_rows=fixture_def.get("truth_total_rows"),
        )

        fixture = cls(
            test_targets=fixture_defs["test_targets"],
            verifier=Verifier.from_args(args),
            assert_passed=fixture_def["assert_passed"],
        )
        return fixture

    @staticmethod
    def unpack_assert_passed(
        assert_passed: bool | dict, *, targets: list | None = None
    ) -> tuple[bool, list] | dict:
        """Unpack assert_passed and return a tuple or dictionary based on the provided targets.

        Parameters
        ----------
            assert_passed : bool, or dict
                A boolean indicating pass/fail status or a dictionary with target-specific statuses.
            targets list, or None
                A list of targets that assert_passed should apply to. If None, the return type is a
                tuple with a bool indicating whether the test is expected to pass and a list of
                parquet file suffixes that are expected to fail. Otherwise, the return type is a dict
                with a key for each target and values indicating pass/fail for the given target.

        Returns
        -------
            tuple[bool, list] | dict:
                - If assert_passed is a boolean:
                    - If targets is None, returns a tuple (assert_passed, []).
                    - Else, returns a dict of {target: assert_passed}.
                - If assert_passed is a dictionary:
                    - If targets is None, assert_passed is expected to contain a single item with
                      key=False and value=list of file suffixes that should have failed. The item
                      is returned as a tuple.
                    - Else, assert_passed is expected to have a key for every target. The
                      assert_passed dict is returned.

        Raises
        ------
            AssertionError: If assert_passed is a dict but it does not have the expected key(s).
        """

        if isinstance(assert_passed, bool):
            if targets is None:
                return assert_passed, []
            return {target: assert_passed for target in targets}

        # assert_passed is a dict

        if targets is None:
            # Expecting a single item with key=False, value=list of file suffixes that should have failed.
            msg = "Unexpected key. There is probably a bug in the fixture definition."
            assert set(assert_passed) == {False}, msg
            return False, assert_passed[False]

        # Expecting one key per target
        msg = "Unexpected set of targets. There is probably a bug in the fixture definition."
        assert set(assert_passed) == set(targets), msg
        return assert_passed
