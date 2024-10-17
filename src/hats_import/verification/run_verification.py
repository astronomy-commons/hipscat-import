"""Run pass/fail checks and generate verification report of existing hats table."""

from hats_import.verification.arguments import VerificationArguments


def run(args):
    """Run verification pipeline."""
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    # implement everything else.
    raise NotImplementedError("Verification not yet implemented.")
