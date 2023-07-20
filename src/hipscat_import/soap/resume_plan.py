"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io
from numpy import frombuffer
from tqdm import tqdm
from hipscat.pixel_math.healpix_pixel import HealpixPixel

from hipscat_import.pipeline_plan import PipelinePlan
from hipscat_import.soap.map_reduce import source_to_object_map
from hipscat_import.soap.arguments import SoapArguments


@dataclass
class SoapPlan(PipelinePlan):
    """Container class for holding the state of each file in the pipeline plan."""

    count_keys: List[Tuple[HealpixPixel, List[HealpixPixel], str]] = field(
        default_factory=list
    )
    """set of files (and job keys) that have yet to be split"""

    COUNTING_LOG_FILE = "counting_log.txt"
    COUNTING_DONE_FILE = "counting_done"
    SOURCE_MAP_FILE = "source_object_map.binary"

    def __init__(self, args: SoapArguments):
        super().__init__(
            resume=args.resume, progress_bar=args.progress_bar, tmp_path=args.tmp_path
        )

        with tqdm(
            total=5, desc="Planning ", disable=not self.progress_bar
        ) as step_progress:
            ## Make sure it's safe to use existing resume state.
            super().safe_to_resume()
            step_progress.update(1)

            ## Validate existing resume state.
            if self.is_counting_done():
                return
            step_progress.update(1)

            source_map_file = file_io.append_paths_to_pointer(self.tmp_path, self.SOURCE_MAP_FILE)
            if file_io.does_file_or_directory_exist(source_map_file):
                ## TODO - read
                source_pixel_map = None
            else:
                source_pixel_map = source_to_object_map(args)
                ## TODO - write
            self._set_sources_to_count(source_pixel_map)
            step_progress.update(1)


    def mark_counting_done(self, counting_key: str):
        """Add counting key to done list."""
        self.write_log_key(self.COUNTING_LOG_FILE, counting_key)

    def is_counting_done(self) -> bool:
        """Are there sources left to count?"""
        return self.done_file_exists(self.COUNTING_DONE_FILE)

    def set_counting_done(self):
        """All sources are done counting."""
        self.touch_done_file(self.COUNTING_DONE_FILE)

    def _set_sources_to_count(self, source_pixel_map):
        """Fetch a triple for each source pixel to join and count.

        Triple contains:

        - source pixel
        - object pixels (healpix pixel with both order and pixel, for aligning and
          neighboring object pixels)
        - source key (string of source order+pixel)
        """
        counted_keys = set(self.read_log_keys(self.COUNTING_LOG_FILE))
        self.count_keys = [
            (hp_pixel, object_pixels, f"{hp_pixel.order}_{hp_pixel.pixel}")
            for hp_pixel, object_pixels in source_pixel_map.items()
            if f"{hp_pixel.order}_{hp_pixel.pixel}" not in counted_keys
        ]
