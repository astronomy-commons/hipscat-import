"""Utility to hold all arguments required throughout partitioning"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from hipscat.catalog.catalog import CatalogInfo
from hipscat.io import FilePointer
from hipscat.pixel_math import hipscat_id

from hipscat_import.catalog.file_readers import InputReader, get_file_reader
from hipscat_import.catalog.resume_plan import ResumePlan
from hipscat_import.runtime_arguments import RuntimeArguments, find_input_paths

# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,too-many-branches,too-few-public-methods


@dataclass
class ImportArguments(RuntimeArguments):
    """Container class for holding partitioning arguments"""

    epoch: str = "J2000"
    """astronomical epoch for the data. defaults to "J2000" """

    catalog_type: str = "object"
    """level of catalog data, object (things in the sky) or source (detections)"""
    input_path: FilePointer | None = None
    """path to search for the input data"""
    input_file_list: List[FilePointer] = field(default_factory=list)
    """can be used instead of input_path to import only specified files"""
    input_paths: List[FilePointer] = field(default_factory=list)
    """resolved list of all files that will be used in the importer"""
    input_storage_options: Union[Dict[Any, Any], None] = None
    """optional dictionary of abstract filesystem credentials for the INPUT."""

    ra_column: str = "ra"
    """column for right ascension"""
    dec_column: str = "dec"
    """column for declination"""
    use_hipscat_index: bool = False
    """use an existing hipscat spatial index as the position, instead of ra/dec"""
    sort_columns: str | None = None
    """column for survey identifier, or other sortable column. if sorting by multiple columns,
    they should be comma-separated. if `add_hipscat_index=True`, this sorting will be used to
    resolve the counter within the same higher-order pixel space"""
    add_hipscat_index: bool = True
    """add the hipscat spatial index field alongside the data"""
    use_schema_file: str | None = None
    """path to a parquet file with schema metadata. this will be used for column
    metadata when writing the files, if specified"""
    resume: bool = True
    """if there are existing intermediate resume files, should we
    read those and continue to create a new catalog where we left off"""
    constant_healpix_order: int = -1
    """healpix order to use when mapping. if this is
    a positive number, this will be the order of all final pixels and we
    will not combine pixels according to the threshold"""
    lowest_healpix_order: int = 0
    """the lowest possible healpix order that we will use for the final 
    catalog partitioning. setting this higher than 0 will prevent creating
    partitions with a large area on the sky."""
    highest_healpix_order: int = 7
    """healpix order to use when mapping. this will
    not necessarily be the order used in the final catalog, as we may combine
    pixels that don't meed the threshold"""
    pixel_threshold: int = 1_000_000
    """maximum number of rows for a single resulting pixel.
    we may combine hierarchically until we near the ``pixel_threshold``"""
    mapping_healpix_order: int = -1
    """healpix order to use when mapping. will be
    ``highest_healpix_order`` unless a positive value is provided for
    ``constant_healpix_order``"""
    debug_stats_only: bool = False
    """do not perform a map reduce and don't create a new
    catalog. generate the partition info"""
    file_reader: InputReader | str | None = None
    """instance of input reader that specifies arguments necessary for reading
    from your input files"""
    resume_plan: ResumePlan | None = None
    """container that handles read/write of log files for this pipeline"""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        """Check existence and consistency of argument values"""
        super()._check_arguments()

        if self.lowest_healpix_order == self.highest_healpix_order:
            self.constant_healpix_order = self.lowest_healpix_order
        if self.constant_healpix_order >= 0:
            check_healpix_order_range(self.constant_healpix_order, "constant_healpix_order")
            self.mapping_healpix_order = self.constant_healpix_order
        else:
            check_healpix_order_range(self.highest_healpix_order, "highest_healpix_order")
            check_healpix_order_range(
                self.lowest_healpix_order, "lowest_healpix_order", upper_bound=self.highest_healpix_order
            )
            if not 100 <= self.pixel_threshold <= 1_000_000_000:
                raise ValueError("pixel_threshold should be between 100 and 1,000,000,000")
            self.mapping_healpix_order = self.highest_healpix_order

        if self.catalog_type not in ("source", "object"):
            raise ValueError("catalog_type should be one of `source` or `object`")

        if (not self.input_path and not self.input_file_list) or (self.input_path and self.input_file_list):
            raise ValueError("exactly one of input_path or input_file_list is required")
        if self.file_reader is None:
            raise ValueError("file_reader is required")
        if isinstance(self.file_reader, str):
            self.file_reader = get_file_reader(self.file_reader)

        if self.use_hipscat_index:
            self.add_hipscat_index = False
            if self.sort_columns:
                raise ValueError("When using _hipscat_index for position, no sort columns should be added")

        # Basic checks complete - make more checks and create directories where necessary
        self.input_paths = find_input_paths(
            self.input_path,
            "**/**.*",
            self.input_file_list,
            storage_options=self.input_storage_options,
        )
        self.resume_plan = ResumePlan(
            resume=self.resume,
            progress_bar=self.progress_bar,
            input_paths=self.input_paths,
            tmp_path=self.resume_tmp,
        )

    def to_catalog_info(self, total_rows) -> CatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "catalog_type": self.catalog_type,
            "total_rows": total_rows,
            "epoch": self.epoch,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
        }
        return CatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        file_reader_info = {"type": self.file_reader}
        if isinstance(self.file_reader, InputReader):
            file_reader_info = self.file_reader.provenance_info()
        return {
            "catalog_name": self.output_artifact_name,
            "epoch": self.epoch,
            "catalog_type": self.catalog_type,
            "input_path": self.input_path,
            "input_paths": self.input_paths,
            "input_file_list": self.input_file_list,
            "ra_column": self.ra_column,
            "dec_column": self.dec_column,
            "use_hipscat_index": self.use_hipscat_index,
            "sort_columns": self.sort_columns,
            "constant_healpix_order": self.constant_healpix_order,
            "lowest_healpix_order": self.lowest_healpix_order,
            "highest_healpix_order": self.highest_healpix_order,
            "pixel_threshold": self.pixel_threshold,
            "mapping_healpix_order": self.mapping_healpix_order,
            "debug_stats_only": self.debug_stats_only,
            "file_reader_info": file_reader_info,
        }


def check_healpix_order_range(
    order, field_name, lower_bound=0, upper_bound=hipscat_id.HIPSCAT_ID_HEALPIX_ORDER
):
    """Helper method to check if the ``order`` is within the range determined by the
    ``lower_bound`` and ``upper_bound``, inclusive.

    Args:
        order (int): healpix order to check
        field_name (str): field name to use in the error message
        lower_bound (int): lower bound of range
        upper_bound (int): upper bound of range
    Raise:
        ValueError: if the order is outside the specified range, or bounds
            are unreasonable.
    """
    if lower_bound < 0:
        raise ValueError("healpix orders must be positive")
    if upper_bound > hipscat_id.HIPSCAT_ID_HEALPIX_ORDER:
        raise ValueError(f"healpix order should be <= {hipscat_id.HIPSCAT_ID_HEALPIX_ORDER}")
    if not lower_bound <= order <= upper_bound:
        raise ValueError(f"{field_name} should be between {lower_bound} and {upper_bound}")
