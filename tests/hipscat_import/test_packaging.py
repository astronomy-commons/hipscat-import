import hipscat_import


def test_hipscat_import_version():
    """Check to see that we can get the hipscat-import version"""
    assert hipscat_import.__version__ is not None
