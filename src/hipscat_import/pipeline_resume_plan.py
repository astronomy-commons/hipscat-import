"""Utility to hold a pipeline's execution and resume plan."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from dask.distributed import as_completed
from hipscat.io import FilePointer, file_io
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from tqdm.auto import tqdm


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

    ORIGINAL_INPUT_PATHS = "input_paths.txt"

    def safe_to_resume(self):
        """Check that we are ok to resume an in-progress pipeline, if one exists.

        Raises:
            ValueError: if the tmp_path already exists and contains some files.
        """
        if file_io.directory_has_contents(self.tmp_path):
            if not self.resume:
                self.clean_resume_files()
            else:
                warnings.warn(
                    f"tmp_path ({self.tmp_path}) contains intermediate files; resuming prior progress."
                )
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
            RuntimeError: if any future returns an error status.
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
                print(f"{stage_name} task {future.key} failed with message:")
                print(future.exception())
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

    def check_original_input_paths(self, input_paths):
        """Validate that we're operating on the same file set as the original pipeline,
        or save the inputs as the originals if not found.

        Args:
            input_paths (list[str]): input paths that will be processed by a pipeline.

        Raises:
            ValueError: if the retrieved file set differs from `input_paths`.
        """
        unique_file_paths = set(input_paths)

        original_input_paths = []

        file_path = file_io.append_paths_to_pointer(self.tmp_path, self.ORIGINAL_INPUT_PATHS)
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                contents = file_handle.readlines()
            contents = [path.strip() for path in contents]
            original_input_paths = set(contents)
        except FileNotFoundError:
            pass

        if len(original_input_paths) == 0:
            file_path = file_io.append_paths_to_pointer(self.tmp_path, self.ORIGINAL_INPUT_PATHS)
            with open(file_path, "w", encoding="utf-8") as file_handle:
                for path in input_paths:
                    file_handle.write(f"{path}\n")
        else:
            if original_input_paths != unique_file_paths:
                raise ValueError("Different file set from resumed pipeline execution.")

        input_paths = list(unique_file_paths)
        input_paths.sort()
        return input_paths


def get_pixel_cache_directory(cache_path, pixel: HealpixPixel):
    """Create a path for intermediate pixel data.

    You can use this over the paths.get_pixel_directory method, as it will include the pixel
    number in the path. Further, it will just *look* different from a real hipscat
    path, so it's clearer that it's a temporary directory::

        {cache_path}/order_{order}/dir_{dir}/pixel_{pixel}/

    Args:
        cache_path (str): root path to cache
        pixel (HealpixPixel): pixel partition data
    """
    return file_io.append_paths_to_pointer(
        cache_path, f"order_{pixel.order}", f"dir_{pixel.dir}", f"pixel_{pixel.pixel}"
    )
