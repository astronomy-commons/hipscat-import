"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from numpy import frombuffer
from tqdm.auto import tqdm

from hipscat_import.pipeline_resume_plan import PipelineResumePlan


@dataclass
class ResumePlan(PipelineResumePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    input_paths: List[FilePointer] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""
    map_files: List[Tuple[str, str]] = field(default_factory=list)
    """list of files (and job keys) that have yet to be mapped"""
    split_keys: List[Tuple[str, str]] = field(default_factory=list)
    """set of files (and job keys) that have yet to be split"""
    destination_pixel_map: Optional[List[Tuple[HealpixPixel, List[HealpixPixel], str]]] = None
    """Fully resolved map of destination pixels to constituent smaller pixels"""

    MAPPING_STAGE = "mapping"
    SPLITTING_STAGE = "splitting"
    REDUCING_STAGE = "reducing"

    HISTOGRAM_BINARY_FILE = "mapping_histogram.binary"
    HISTOGRAMS_DIR = "histograms"

    def __post_init__(self):
        """Initialize the plan."""
        self.gather_plan()

    def gather_plan(self):
        """Initialize the plan."""
        with tqdm(
            total=5, desc=self.get_formatted_stage_name("Planning"), disable=not self.progress_bar
        ) as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            step_progress.update(1)

            ## Validate existing resume state.
            ## - if a later stage is complete, the earlier stages should be complete too.
            mapping_done = self.is_mapping_done()
            splitting_done = self.is_splitting_done()
            reducing_done = self.is_reducing_done()

            if reducing_done and (not mapping_done or not splitting_done):
                raise ValueError("mapping and splitting must be complete before reducing")
            if splitting_done and not mapping_done:
                raise ValueError("mapping must be complete before splitting")
            step_progress.update(1)

            ## Validate that we're operating on the same file set as the previous instance.
            self.input_paths = self.check_original_input_paths(self.input_paths)
            step_progress.update(1)

            ## Gather keys for execution.
            if not mapping_done:
                self.map_files = self.get_remaining_map_keys()
            if not splitting_done:
                self.split_keys = self.get_remaining_split_keys()
            ## We don't pre-gather the plan for the reducing keys.
            ## It requires the full destination pixel map.
            step_progress.update(1)
            ## Go ahead and create our directories for storing resume files.
            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAMS_DIR),
                exist_ok=True,
            )
            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.SPLITTING_STAGE),
                exist_ok=True,
            )
            file_io.make_directory(
                file_io.append_paths_to_pointer(self.tmp_path, self.REDUCING_STAGE),
                exist_ok=True,
            )
            step_progress.update(1)

    def get_remaining_map_keys(self):
        """Gather remaining keys, dropping successful mapping tasks from histogram names.

        Returns:
            list of mapping keys *not* found in files like /resume/path/mapping_key.binary
        """
        prefix = file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAMS_DIR)
        mapped_keys = self.get_keys_from_file_names(prefix, ".binary")
        return [
            (f"map_{i}", file_path)
            for i, file_path in enumerate(self.input_paths)
            if f"map_{i}" not in mapped_keys
        ]

    def read_histogram(self, healpix_order):
        """Return histogram with healpix_order'd shape

        - Try to find a combined histogram
        - Otherwise, combine histograms from partials
        - Otherwise, return an empty histogram

        """
        full_histogram = pixel_math.empty_histogram(healpix_order)

        ## Look for the single combined histogram file.
        file_name = file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAM_BINARY_FILE)
        if file_io.does_file_or_directory_exist(file_name):
            with open(file_name, "rb") as file_handle:
                return frombuffer(file_handle.read(), dtype=np.int64)

        ## Otherwise:
        # - read all the partial histograms
        # - combine into a single histogram
        # - write out as a single histogram for future reads
        # - remove all partial histograms
        remaining_map_files = self.get_remaining_map_keys()
        if len(remaining_map_files) > 0:
            raise RuntimeError(f"{len(remaining_map_files)} map stages did not complete successfully.")
        histogram_files = file_io.find_files_matching_path(self.tmp_path, self.HISTOGRAMS_DIR, "**.binary")
        for file_name in histogram_files:
            with open(file_name, "rb") as file_handle:
                full_histogram = np.add(full_histogram, frombuffer(file_handle.read(), dtype=np.int64))

        file_name = file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAM_BINARY_FILE)
        with open(file_name, "wb+") as file_handle:
            file_handle.write(full_histogram.data)
        file_io.remove_directory(
            file_io.append_paths_to_pointer(self.tmp_path, self.HISTOGRAMS_DIR),
            ignore_errors=True,
        )
        return full_histogram

    @classmethod
    def write_partial_histogram(cls, tmp_path, mapping_key: str, histogram):
        """Write partial histogram to a special intermediate directory

        Args:
            tmp_path (str): where to write intermediate resume files.
            mapping_key (str): unique string for each mapping task (e.g. "map_57")
            histogram (np.array): one-dimensional numpy array of long integers where
                the value at each index corresponds to the number of objects found at
                the healpix pixel.
        """
        file_io.make_directory(
            file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR),
            exist_ok=True,
        )
        file_name = file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR, f"{mapping_key}.binary")
        with open(file_name, "wb+") as file_handle:
            file_handle.write(histogram.data)

    def get_remaining_split_keys(self):
        """Gather remaining keys, dropping successful split tasks from done file names.

        Returns:
            list of splitting keys *not* found in files like /resume/path/split_key.done
        """
        split_keys = set(self.read_done_keys(self.SPLITTING_STAGE))
        return [
            (f"split_{i}", file_path)
            for i, file_path in enumerate(self.input_paths)
            if f"split_{i}" not in split_keys
        ]

    @classmethod
    def splitting_key_done(cls, tmp_path, splitting_key: str):
        """Mark a single splitting task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            splitting_key (str): unique string for each splitting task (e.g. "split_57")
        """
        cls.touch_key_done_file(tmp_path, cls.SPLITTING_STAGE, splitting_key)

    @classmethod
    def reducing_key_done(cls, tmp_path, reducing_key: str):
        """Mark a single reducing task as done

        Args:
            tmp_path (str): where to write intermediate resume files.
            reducing_key (str): unique string for each reducing task (e.g. "3_57")
        """
        cls.touch_key_done_file(tmp_path, cls.REDUCING_STAGE, reducing_key)

    def wait_for_mapping(self, futures):
        """Wait for mapping futures to complete."""
        self.wait_for_futures(futures, self.MAPPING_STAGE)
        remaining_map_items = self.get_remaining_map_keys()
        if len(remaining_map_items) > 0:
            raise RuntimeError("some map stages did not complete successfully.")
        self.touch_stage_done_file(self.MAPPING_STAGE)

    def is_mapping_done(self) -> bool:
        """Are there files left to map?"""
        return self.done_file_exists(self.MAPPING_STAGE)

    def wait_for_splitting(self, futures):
        """Wait for splitting futures to complete."""
        self.wait_for_futures(futures, self.SPLITTING_STAGE)
        remaining_split_items = self.get_remaining_split_keys()
        if len(remaining_split_items) > 0:
            raise RuntimeError(f"{len(remaining_split_items)} split stages did not complete successfully.")
        self.touch_stage_done_file(self.SPLITTING_STAGE)

    def is_splitting_done(self) -> bool:
        """Are there files left to split?"""
        return self.done_file_exists(self.SPLITTING_STAGE)

    def get_reduce_items(self, destination_pixel_map=None):
        """Fetch a triple for each partition to reduce.

        Triple contains:

        - destination pixel (healpix pixel with both order and pixel)
        - source pixels (list of pixels at mapping order)
        - reduce key (string of destination order+pixel)

        """
        reduced_keys = set(self.read_done_keys(self.REDUCING_STAGE))
        if destination_pixel_map is None:
            destination_pixel_map = self.destination_pixel_map
        elif self.destination_pixel_map is None:
            self.destination_pixel_map = destination_pixel_map
        if self.destination_pixel_map is None:
            raise RuntimeError("destination pixel map not provided for progress tracking.")
        reduce_items = [
            (hp_pixel, source_pixels, f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel, source_pixels in destination_pixel_map.items()
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in reduced_keys
        ]
        return reduce_items

    def is_reducing_done(self) -> bool:
        """Are there partitions left to reduce?"""
        return self.done_file_exists(self.REDUCING_STAGE)

    def wait_for_reducing(self, futures):
        """Wait for reducing futures to complete."""
        self.wait_for_futures(futures, self.REDUCING_STAGE)
        remaining_reduce_items = self.get_reduce_items()
        if len(remaining_reduce_items) > 0:
            raise RuntimeError(f"{len(remaining_reduce_items)} reduce stages did not complete successfully.")
        self.touch_stage_done_file(self.REDUCING_STAGE)
