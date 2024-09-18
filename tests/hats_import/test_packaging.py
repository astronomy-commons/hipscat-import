import hats_import


def test_hats_import_version():
    """Check to see that we can get the hats-import version"""
    assert hats_import.__version__ is not None
