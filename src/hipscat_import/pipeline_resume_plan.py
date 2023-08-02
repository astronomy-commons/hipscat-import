"""Utility to hold a pipeline's execution and resume plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from dask.distributed import as_completed
from hipscat.io import FilePointer, file_io
from tqdm import tqdm


@dataclass
class PipelineResumePlan:
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
        """Check that we are ok to resume an in-progress pipeline, if one exists.

        Raises:
            ValueError: if the tmp_path already exists and contains some files.
        """
        if not self.resume:
            if file_io.directory_has_contents(self.tmp_path):
                raise ValueError(
                    f"tmp_path ({self.tmp_path}) contains intermediate files."
                    " choose a different directory or use --resume flag"
                )
        file_io.make_directory(self.tmp_path, exist_ok=True)

    def done_file_exists(self, stage_name):
        """Is there a file at a given path?
        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Returns:
            boolean, True if the done file exists at tmp_path. False otherwise.
        """
        return file_io.does_file_or_directory_exist(
            file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_done")
        )

    def touch_done_file(self, stage_name):
        """Touch (create) a done file at the given path.
        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        """
        Path(file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_done")).touch()

    def read_log_keys(self, stage_name):
        """Read a resume log file, containing timestamp and keys.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Return:
            List[str] - all keys found in the log file
        """
        file_path = file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_log.txt")
        if file_io.does_file_or_directory_exist(file_path):
            mapping_log = pd.read_csv(
                file_path,
                delimiter="\t",
                header=None,
                names=["time", "key"],
            )
            return mapping_log["key"].tolist()
        return []

    def write_log_key(self, stage_name, key):
        """Append a tab-delimited line to the file with the current timestamp and provided key

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
            key(str): single key to write
        """
        file_path = file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_log.txt")
        with open(file_path, "a", encoding="utf-8") as mapping_log:
            mapping_log.write(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\t{key}\n')

    def clean_resume_files(self):
        """Remove all intermediate files created in execution."""
        file_io.remove_directory(self.tmp_path, ignore_errors=True)

    def wait_for_futures(self, futures, stage_name):
        """Wait for collected futures to complete.

        As each future completes, read the task key and write to the log file.
        If all tasks complete successfully, touch the done file. Otherwise, raise an error.

        Args:
            futures(List[future]): collected futures
            stage_name(str): name of the stage (e.g. mapping, reducing)
        """
        some_error = False
        for future in tqdm(
            as_completed(futures),
            desc=stage_name,
            total=len(futures),
            disable=(not self.progress_bar),
        ):
            if future.status == "error":  # pragma: no cover
                some_error = True
            else:
                self.write_log_key(stage_name, future.key)
        if some_error:  # pragma: no cover
            raise RuntimeError(f"Some {stage_name} stages failed. See logs for details.")
        self.touch_done_file(stage_name)
