"""Utility to hold a pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from hipscat.io import FilePointer, file_io


@dataclass
class PipelinePlan:
    """Container class for holding the state of pipeline plan."""

    tmp_path: FilePointer
    """path for any intermediate files"""
    resume: bool = False
    """if there are existing intermediate resume files, should we
    read those and continue to run pipeline where we left off"""
    progress_bar: bool = True
    """if true, a tqdm progress bar will be displayed for user
    feedback of planning progress"""

    def safe_to_resume(self):
        """Check that we are ok to resume an in-progress pipeline, if one exists"""
        if not self.resume:
            if file_io.directory_has_contents(self.tmp_path):
                raise ValueError(
                    f"tmp_path ({self.tmp_path}) contains intermediate files."
                    " choose a different directory or use --resume flag"
                )
        file_io.make_directory(self.tmp_path, exist_ok=True)

    def done_file_exists(self, file_name):
        """Is there a file at a given path?

        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.
        """
        return file_io.does_file_or_directory_exist(
            file_io.append_paths_to_pointer(self.tmp_path, file_name)
        )

    def touch_done_file(self, file_name):
        """Touch (create) a done file at the given path.

        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.
        """
        Path(file_io.append_paths_to_pointer(self.tmp_path, file_name)).touch()

    def read_log_keys(self, file_name):
        """Read a resume log file, containing timestamp and keys."""
        file_path = file_io.append_paths_to_pointer(self.tmp_path, file_name)
        if file_io.does_file_or_directory_exist(file_path):
            mapping_log = pd.read_csv(
                file_path,
                delimiter="\t",
                header=None,
                names=["time", "key"],
            )
            return mapping_log["key"].tolist()
        return []

    def write_log_key(self, file_name, key):
        """Append a tab-delimited line to the file with the current timestamp and provided key"""
        file_path = file_io.append_paths_to_pointer(self.tmp_path, file_name)
        with open(file_path, "a", encoding="utf-8") as mapping_log:
            mapping_log.write(
                f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\t{key}\n'
            )

    def clean_resume_files(self):
        """Remove all intermediate files created in execution."""
        file_io.remove_directory(self.tmp_path, ignore_errors=True)
