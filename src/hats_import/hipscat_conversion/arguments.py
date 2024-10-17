"""Utility to hold all arguments required throughout hipscat -> hats conversion"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hats.io import file_io
from upath import UPath

from hats_import.runtime_arguments import RuntimeArguments


@dataclass
class ConversionArguments(RuntimeArguments):
    """Data class for holding conversion arguments. Mostly just inheriting from RuntimeArguments"""

    ## Input
    input_catalog_path: str | Path | UPath | None = None

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        self.input_catalog_path = file_io.get_upath(self.input_catalog_path)
