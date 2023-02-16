"""Main method to enable command line execution"""

import sys

import hipscat_import.run_import as runner
from hipscat_import.command_line_arguments import parse_command_line

if __name__ == "__main__":
    runner.run(parse_command_line(sys.argv[1:]))
