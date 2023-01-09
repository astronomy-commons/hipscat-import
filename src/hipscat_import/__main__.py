"""Main method to enable command line execution"""

import sys

import hipscat_import.dask_map_reduce as runner
from hipscat_import.arguments import ImportArguments

if __name__ == "__main__":
    args = ImportArguments()
    args.from_command_line(sys.argv[1:])
    runner.run(args)
