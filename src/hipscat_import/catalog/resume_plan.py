"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io
from numpy import frombuffer
from tqdm import tqdm


@dataclass
class ResumePlan:
    """Container class for holding the state of each file in the pipeline plan."""

    tmp_path: FilePointer
    """path for any intermediate files"""
    resume: bool = False
    """if there are existing intermediate resume files, should we
    read those and continue to create a new catalog where we left off"""
    progress_bar: bool = True
    """if true, a tqdm progress bar will be displayed for user
    feedback of planning progress"""
    input_paths: List[FilePointer] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""
    map_files: List[Tuple[str, str]] = field(default_factory=list)
    """list of files (and job keys) that have yet to be mapped"""
    split_keys: List[Tuple[str, str]] = field(default_factory=list)
    """set of files (and job keys) that have yet to be split"""

    MAPPING_LOG_FILE = "mapping_log.txt"
    SPLITTING_LOG_FILE = "splitting_log.txt"
    REDUCING_LOG_FILE = "reducing_log.txt"

    HISTOGRAM_BINARY_FILE = "mapping_histogram.binary"
    HISTOGRAMS_DIR = "histograms"

    MAPPING_DONE_FILE = "mapping_done"
    SPLITTING_DONE_FILE = "splitting_done"
    REDUCING_DONE_FILE = "reducing_done"

    def __post_init__(self):
        """Initialize the plan."""
        self.gather_plan()

    def gather_plan(self):
        """Initialize the plan."""
        with tqdm(
            total=4, desc="Planning ", disable=not self.progress_bar
        ) as step_progress:
            ## Make sure it's safe to use existing resume state.
            if not self.resume:
                if file_io.directory_has_contents(self.tmp_path):
                    raise ValueError(
                        f"tmp_path ({self.tmp_path}) contains intermediate files."
                        " choose a different directory or use --resume flag"
                    )
            file_io.make_directory(self.tmp_path, exist_ok=True)
            step_progress.update(1)

            ## Validate existing resume state.
            ## - if a later stage is complete, the earlier stages should be complete too.
            mapping_done = self.is_mapping_done()
            splitting_done = self.is_splitting_done()
            reducing_done = self.is_reducing_done()

            if reducing_done and (not mapping_done or not splitting_done):
                raise ValueError(
                    "mapping and splitting must be complete before reducing"
                )
            if splitting_done and not mapping_done:
                raise ValueError("mapping must be complete before splitting")
            step_progress.update(1)

            ## Gather keys for execution.
            self.input_paths.sort()
            step_progress.update(1)
            if not mapping_done:
                mapped_keys = set(self._read_log_keys(self.MAPPING_LOG_FILE))
                self.map_files = [
                    (f"map_{i}", file_path)
                    for i, file_path in enumerate(self.input_paths)
                    if f"map_{i}" not in mapped_keys
                ]
            if not splitting_done:
                split_keys = set(self._read_log_keys(self.SPLITTING_LOG_FILE))
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
        ."""
        full_histogram = pixel_math.empty_histogram(healpix_order)

        ## Look for the single combined histogram file.
        file_name = file_io.append_paths_to_pointer(
            self.tmp_path, self.HISTOGRAM_BINARY_FILE
        )
        if file_io.does_file_or_directory_exist(file_name):
            with open(file_name, "rb") as file_handle:
                return frombuffer(file_handle.read(), dtype=np.int64)

        ## Otherwise:
        # - read all the partial histograms
        # - combine into a single histogram
        # - write out as a single histogram for future reads
        # - remove all partial histograms
        histogram_files = file_io.find_files_matching_path(
            self.tmp_path, self.HISTOGRAMS_DIR, "**.binary"
        )
        for file_name in histogram_files:
            with open(file_name, "rb") as file_handle:
                full_histogram = np.add(
                    full_histogram, frombuffer(file_handle.read(), dtype=np.int64)
                )

        file_name = file_io.append_paths_to_pointer(
            self.tmp_path, self.HISTOGRAM_BINARY_FILE
        )
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

        file_name = file_io.append_paths_to_pointer(
            tmp_path, cls.HISTOGRAMS_DIR, f"{mapping_key}.binary"
        )
        with open(file_name, "wb+") as file_handle:
            file_handle.write(histogram.data)

    def mark_mapping_done(self, mapping_key: str):
        """Add mapping key to done list and update raw histogram"""
        self._write_log_key(self.MAPPING_LOG_FILE, mapping_key)

    def is_mapping_done(self) -> bool:
        """Are there files left to map?"""
        return self._done_file_exists(self.MAPPING_DONE_FILE)

    def set_mapping_done(self):
        """All files are done mapping."""
        self._touch_done_file(self.MAPPING_DONE_FILE)

    def mark_splitting_done(self, splitting_key: str):
        """Add splitting key to done list"""
        self._write_log_key(self.SPLITTING_LOG_FILE, splitting_key)

    def is_splitting_done(self) -> bool:
        """Are there files left to split?"""
        return self._done_file_exists(self.SPLITTING_DONE_FILE)

    def set_splitting_done(self):
        """All files are done splitting."""
        self._touch_done_file(self.SPLITTING_DONE_FILE)

    def get_reduce_items(self, destination_pixel_map):
        """Fetch a triple for each partition to reduce.

        Triple contains:

        - destination pixel (healpix pixel with both order and pixel)
        - source pixels (list of pixels at mapping order)
        - reduce key (string of destination order+pixel)
        """
        reduced_keys = set(self._read_log_keys(self.REDUCING_LOG_FILE))
        reduce_items = [
            (hp_pixel, source_pixels, f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel, source_pixels in destination_pixel_map.items()
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in reduced_keys
        ]
        return reduce_items

    def mark_reducing_done(self, reducing_key: str):
        """Add reducing key to done list"""
        self._write_log_key(self.REDUCING_LOG_FILE, reducing_key)

    def is_reducing_done(self) -> bool:
        """Are there partitions left to reduce?"""
        return self._done_file_exists(self.REDUCING_DONE_FILE)

    def set_reducing_done(self):
        """All partitions are done reducing."""
        self._touch_done_file(self.REDUCING_DONE_FILE)

    def clean_resume_files(self):
        """Remove all intermediate files created in execution."""
        file_io.remove_directory(self.tmp_path, ignore_errors=True)

    #####################################################################
    ###                     Helper methods                            ###
    #####################################################################

    def _done_file_exists(self, file_name):
        return file_io.does_file_or_directory_exist(
            file_io.append_paths_to_pointer(self.tmp_path, file_name)
        )

    def _touch_done_file(self, file_name):
        Path(file_io.append_paths_to_pointer(self.tmp_path, file_name)).touch()

    def _read_log_keys(self, file_name):
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

    def _write_log_key(self, file_name, key):
        """Append a tab-delimited line to the file with the current timestamp and provided key"""
        file_path = file_io.append_paths_to_pointer(self.tmp_path, file_name)
        with open(file_path, "a", encoding="utf-8") as mapping_log:
            mapping_log.write(
                f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\t{key}\n'
            )
