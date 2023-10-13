"""Utility to hold a pipeline's execution and resume plan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dask.distributed import as_completed
from hipscat.io import FilePointer, file_io
from tqdm import tqdm


@dataclass
class PipelineResumePlan:
    """Container class for holding the state of pipeline plan."""

    tmp_path: FilePointer
    """path for any intermediate files"""
    resume: bool = True
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
        if file_io.directory_has_contents(self.tmp_path):
            if not self.resume:
                raise ValueError(
                    f"tmp_path ({self.tmp_path}) contains intermediate files."
                    " choose a different directory or use --resume flag"
                )
            print(f"tmp_path ({self.tmp_path}) contains intermediate files. resuming prior progress.")
        else:
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

    def touch_stage_done_file(self, stage_name):
        """Touch (create) a done file for a whole pipeline stage.
        For a done file, the existence of the file is the only signal needed to indicate
        a pipeline segment is complete.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        """
        Path(file_io.append_paths_to_pointer(self.tmp_path, f"{stage_name}_done")).touch()

    @classmethod
    def touch_key_done_file(cls, tmp_path, stage_name, key):
        """Touch (create) a done file for a single key, within a pipeline stage.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
            stage_name(str): name of the stage (e.g. mapping, reducing)
            key (str): unique string for each task (e.g. "map_57")
        """
        Path(file_io.append_paths_to_pointer(tmp_path, stage_name, f"{key}_done")).touch()

    def read_done_keys(self, stage_name):
        """Inspect the stage's directory of done files, fetching the keys from done file names.

        Args:
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Return:
            List[str] - all keys found in done directory
        """
        prefix = file_io.append_paths_to_pointer(self.tmp_path, stage_name)
        return self.get_keys_from_file_names(prefix, "_done")

    @staticmethod
    def get_keys_from_file_names(directory, extension):
        """Gather keys for successful tasks from result file names.

        Args:
            directory: where to look for result files. this is NOT a recursive lookup
            extension (str): file suffix to look for and to remove from all file names.
                if you expect a file like "map_01.csv", extension should be ".csv"

        Returns:
            list of keys taken from files like /resume/path/{key}{extension}
        """
        result_files = file_io.find_files_matching_path(directory, f"**{extension}")
        keys = []
        for file_path in result_files:
            result_file_name = file_io.get_basename_from_filepointer(file_path)
            match = re.match(r"(.*)" + extension, str(result_file_name))
            keys.append(match.group(1))
        return keys

    def clean_resume_files(self):
        """Remove all intermediate files created in execution."""
        file_io.remove_directory(self.tmp_path, ignore_errors=True)

    def wait_for_futures(self, futures, stage_name):
        """Wait for collected futures to complete.

        As each future completes, check the returned status.

        Args:
            futures(List[future]): collected futures
            stage_name(str): name of the stage (e.g. mapping, reducing)
        Raises:
            RuntimeError if any future returns an error status.
        """
        some_error = False
        formatted_stage_name = self.get_formatted_stage_name(stage_name)
        for future in tqdm(
            as_completed(futures),
            desc=formatted_stage_name,
            total=len(futures),
            disable=(not self.progress_bar),
        ):
            if future.status == "error":
                some_error = True
        if some_error:
            raise RuntimeError(f"Some {stage_name} stages failed. See logs for details.")

    @staticmethod
    def get_formatted_stage_name(stage_name) -> str:
        """Create a stage name of consistent minimum length. Ensures that the tqdm
        progress bars can line up nicely when multiple stages must run.

        Args:
            stage_name (str): name of the stage (e.g. mapping, reducing)
        """
        if stage_name is None or len(stage_name) == 0:
            stage_name = "progress"

        return f"{stage_name.capitalize(): <10}"
