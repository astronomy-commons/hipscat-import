from hipscat_import.cross_match.macauff_arguments import MacauffArguments

# pylint: disable=unused-argument


def run(args, client):
    """run macauff cross-match import pipeline"""
    if not args:
        raise TypeError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise TypeError("args must be type MacauffArguments")

    raise NotImplementedError("macauff pipeline not implemented yet.")
