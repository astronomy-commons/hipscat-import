"""Utility to hold all arguments required throughout verification pipeline"""

from __future__ import annotations

from pathlib import Path

import attrs
from upath import UPath

# from hipscat_import.runtime_arguments import RuntimeArguments


def _dir_exists(instance: VerificationArguments, attribute: attrs.Attribute, value: UPath):
    """This function will be used as a validator for attributes of VerificationArguments."""
    if not value.is_dir():
        raise ValueError(f"{attribute.name} must be an existing directory")


def _path_exists(instance: VerificationArguments, attribute: attrs.Attribute, value: UPath):
    """This function will be used as a validator for attributes of VerificationArguments."""
    if not value.exists():
        raise ValueError(f"{attribute.name} must be an existing file or directory")


@attrs.define(kw_only=True)
class VerificationArguments:
    """Container for verification arguments."""

    input_catalog_path: str | Path | UPath = attrs.field(converter=UPath, validator=_dir_exists)
    """Path to an existing catalog that will be inspected. This must be a directory
    containing the Parquet dataset and metadata sidecars."""
    output_path: str | Path | UPath = attrs.field(converter=UPath)
    """Base path where output files should be written."""
    output_report_filename: str = attrs.field(factory=lambda: "verifier_results.csv")
    """Filename for the verification report that will be generated."""
    output_distributions_filename: str = attrs.field(factory=lambda: "field_distributions.csv")
    """Filename for the field distributions that will be calculated."""
    truth_total_rows: int | None = attrs.field(default=None)
    """Total number of rows expected in this catalog."""
    truth_schema: str | Path | UPath | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(UPath),
        validator=attrs.validators.optional(_path_exists),
    )
    """Path to a Parquet file or dataset containing the expected schema.
    If you provided the 'use_schema_file' argument when importing the catalog, use the same value here.
    If not provided, the catalog's _common_metadata file will be used as the source of truth.
    """

    # [FIXME] Connect this with RuntimeArguments.provenance_info. Even then, does this ever get written to file?
    def additional_runtime_provenance_info(self) -> dict:
        return {"pipeline": "verification pipeline", **{k: str(v) for k, v in vars(self).items()}}
