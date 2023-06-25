"""Utility to hold the file-level pipeline execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from hipscat.io import FilePointer, file_io

import hipscat_import.catalog.resume_files as resume


@dataclass
class ResumePlan:
    """Container class for holding the state of each file in the pipeline plan."""

    resume: bool = False
    progress_bar: bool = True
    input_paths: List[FilePointer] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""
    tmp_path: str = ""
    map_files: List[str] = field(default_factory=list)
    split_keys: List[(str, str)] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the plan."""

        ## Make sure it's safe to use existing resume state.
        if not self.resume:
            if file_io.directory_has_contents(self.tmp_path):
                raise ValueError(
                    f"tmp_path ({self.tmp_path}) contains intermediate files."
                    " choose a different directory or use --resume flag"
                )
        file_io.make_directory(self.tmp_path, exist_ok=True)

        ## Validate existing resume state.
        ## - if a later stage is complete, the earlier stages should be complete too.
        mapping_done = self.is_mapping_done()
        splitting_done = self.is_splitting_done()
        reducing_done = self.is_reducing_done()

        if reducing_done and (not mapping_done or not splitting_done):
            raise ValueError("mapping and splitting must be complete before reducing")
        if splitting_done and not mapping_done:
            raise ValueError("mapping must be complete before splitting")

        ## Gather keys for execution.
        self.input_paths.sort()
        for test_path in self.input_paths:
            if not file_io.does_file_or_directory_exist(test_path):
                raise FileNotFoundError(f"{test_path} not found on local storage")

        if not mapping_done:
            mapped_paths = set(resume.read_mapping_keys(self.tmp_path))
            self.map_files = [
                file_path
                for file_path in self.input_paths
                if f"map_{file_path}" not in mapped_paths
            ]
        if not splitting_done:
            split_keys = set(resume.read_splitting_keys(self.tmp_path))
            self.split_keys = [
                (f"split_{i}", file_path)
                for i, file_path in enumerate(self.input_paths)
            ]
            print(self.split_keys)
            self.split_keys = [
                (key, file) for (key, file) in self.split_keys if key not in split_keys
            ]
        if not reducing_done:
            ...

    def mark_mapping_done(self, mapping_key: str, histogram):
        """Add mapping key to done list and update raw histogram"""
        resume.write_mapping_start_key(self.tmp_path, mapping_key)
        resume.write_histogram(self.tmp_path, histogram)
        resume.write_mapping_done_key(self.tmp_path, mapping_key)

    def is_mapping_done(self) -> bool:
        """Are there files left to map?"""
        return resume.is_mapping_done(self.tmp_path)

    def set_mapping_done(self):
        """All files are done mapping."""
        resume.set_mapping_done(self.tmp_path)

    def get_split_keys(self) -> list[str]:
        """Get job keys for remaining splitting jobs."""
        return self.split_keys

    def mark_splitting_done(self, splitting_key: str):
        """Add splitting key to done list"""
        resume.write_splitting_done_key(self.tmp_path, splitting_key)

    def is_splitting_done(self) -> bool:
        """Are there files left to split?"""
        return resume.is_splitting_done(self.tmp_path)

    def set_splitting_done(self):
        """All files are done splitting."""
        resume.set_splitting_done(self.tmp_path)

    def is_reducing_done(self) -> bool:
        """Are there partitions left to reduce?"""
        return resume.is_reducing_done(self.tmp_path)
