"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io
from numpy import frombuffer
from tqdm import tqdm

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

    MAPPING_STAGE = "mapping"
    SPLITTING_STAGE = "splitting"
    REDUCING_STAGE = "reducing"

    ORIGINAL_INPUT_PATHS = "input_paths.txt"

    HISTOGRAM_BINARY_FILE = "mapping_histogram.binary"
    HISTOGRAMS_DIR = "histograms"

    def __post_init__(self):
        """Initialize the plan."""
        self.gather_plan()

    def gather_plan(self):
        """Initialize the plan."""
        with tqdm(total=5, desc="Planning ", disable=not self.progress_bar) as step_progress:
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
            unique_file_paths = set(self.input_paths)
            self.input_paths = list(unique_file_paths)
            self.input_paths.sort()
            original_input_paths = set(self.read_log_keys(self.ORIGINAL_INPUT_PATHS))
            if not original_input_paths:
                for input_path in self.input_paths:
                    self.write_log_key(self.ORIGINAL_INPUT_PATHS, input_path)
            else:
                if original_input_paths != unique_file_paths:
                    raise ValueError("Different file set from resumed pipeline execution.")
            step_progress.update(1)

            ## Gather keys for execution.
            step_progress.update(1)
            if not mapping_done:
                mapped_keys = set(self.read_log_keys(self.MAPPING_STAGE))
                self.map_files = [
                    (f"map_{i}", file_path)
                    for i, file_path in enumerate(self.input_paths)
                    if f"map_{i}" not in mapped_keys
                ]
            if not splitting_done:
                split_keys = set(self.read_log_keys(self.SPLITTING_STAGE))
                self.split_keys = [
                    (f"split_{i}", file_path)
                    for i, file_path in enumerate(self.input_paths)
                    if f"split_{i}" not in split_keys
                ]
            ## We don't pre-gather the plan for the reducing keys.
            ## It requires the full destination pixel map.
            step_progress.update(1)

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
        """Write partial histogram to a special intermediate directory"""
        file_io.make_directory(
            file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR),
            exist_ok=True,
        )

        file_name = file_io.append_paths_to_pointer(tmp_path, cls.HISTOGRAMS_DIR, f"{mapping_key}.binary")
        with open(file_name, "wb+") as file_handle:
            file_handle.write(histogram.data)

    def wait_for_mapping(self, futures):
        """Wait for mapping futures to complete."""
        self.wait_for_futures(futures, self.MAPPING_STAGE)

    def is_mapping_done(self) -> bool:
        """Are there files left to map?"""
        return self.done_file_exists(self.MAPPING_STAGE)

    def wait_for_splitting(self, futures):
        """Wait for splitting futures to complete."""
        self.wait_for_futures(futures, self.SPLITTING_STAGE)

    def is_splitting_done(self) -> bool:
        """Are there files left to split?"""
        return self.done_file_exists(self.SPLITTING_STAGE)

    def get_reduce_items(self, destination_pixel_map):
        """Fetch a triple for each partition to reduce.

        Triple contains:

        - destination pixel (healpix pixel with both order and pixel)
        - source pixels (list of pixels at mapping order)
        - reduce key (string of destination order+pixel)

        """
        reduced_keys = set(self.read_log_keys(self.REDUCING_STAGE))
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
