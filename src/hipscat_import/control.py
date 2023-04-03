"""Flow control and scripting entry points."""
import sys

import hipscat_import.catalog.run_import as runner
from hipscat_import.catalog.command_line_arguments import parse_command_line


def main():
    """Wrapper of main for setuptools."""
    runner.run(parse_command_line(sys.argv[1:]))
