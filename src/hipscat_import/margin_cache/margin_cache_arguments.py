from dataclasses import dataclass, field
from typing import List, Optional

import hipscat.pixel_math.healpix_shim as hp
from hipscat.catalog import Catalog
from hipscat.catalog.margin_cache.margin_cache_catalog_info import MarginCacheCatalogInfo
from hipscat.io.validation import is_valid_catalog
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from upath import UPath

from hipscat_import.runtime_arguments import RuntimeArguments


@dataclass
class MarginCacheArguments(RuntimeArguments):
    """Container for margin cache generation arguments"""

    margin_threshold: float = 5.0
    """the size of the margin cache boundary, given in arcseconds. setting the
    `margin_threshold` to be greater than the resolution of a `margin_order`
    healpixel will result in a warning, as this may lead to data loss."""
    margin_order: int = -1
    """the order of healpixels that will be used to constrain the margin data before
    doing more precise boundary checking. this value must be greater than the highest
    order of healpix partitioning in the source catalog. if `margin_order` is left
    default or set to -1, then the `margin_order` will be set dynamically to the
    highest partition order plus 1."""
    fine_filtering: bool = True
    """should we perform the precise boundary checking? if false, some results may be
    greater than `margin_threshold` away from the border (but within `margin_order`)."""
    delete_intermediate_parquet_files: bool = True
    """should we delete the smaller intermediate parquet files generated in the
    splitting stage, once the relevant reducing stage is complete?"""
    delete_resume_log_files: bool = True
    """should we delete task-level done files once each stage is complete?
    if False, we will keep all done marker files at the end of the pipeline."""

    input_catalog_path: Optional[UPath] = None
    """the path to the hipscat-formatted input catalog."""
    debug_filter_pixel_list: List[HealpixPixel] = field(default_factory=list)
    """debug setting. if provided, we will first filter the catalog to the pixels
    provided. this can be useful for creating a margin over a subset of a catalog."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not self.input_catalog_path:
            raise ValueError("input_catalog_path is required")
        if not is_valid_catalog(self.input_catalog_path):
            raise ValueError("input_catalog_path not a valid catalog")

        self.catalog = Catalog.read_from_hipscat(self.input_catalog_path)
        if len(self.debug_filter_pixel_list) > 0:
            self.catalog = self.catalog.filter_from_pixel_list(self.debug_filter_pixel_list)
            if len(self.catalog.get_healpix_pixels()) == 0:
                raise ValueError("debug_filter_pixel_list has created empty catalog")

        highest_order = int(self.catalog.partition_info.get_highest_order())

        if self.margin_order < 0:
            self.margin_order = hp.margin2order(margin_thr_arcmin=self.margin_threshold / 60.0)

        if self.margin_order < highest_order + 1:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )

        margin_pixel_nside = hp.order2nside(self.margin_order)
        margin_pixel_avgsize = hp.nside2resol(margin_pixel_nside, arcmin=True)
        margin_pixel_mindist = hp.avgsize2mindist(margin_pixel_avgsize)
        if margin_pixel_mindist * 60.0 < self.margin_threshold:
            raise ValueError("margin pixels must be larger than margin_threshold")

    def to_catalog_info(self, total_rows) -> MarginCacheCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_artifact_name,
            "total_rows": total_rows,
            "catalog_type": "margin",
            "epoch": self.catalog.catalog_info.epoch,
            "ra_column": self.catalog.catalog_info.ra_column,
            "dec_column": self.catalog.catalog_info.dec_column,
            "primary_catalog": self.input_catalog_path,
            "margin_threshold": self.margin_threshold,
        }
        return MarginCacheCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "input_catalog_path": self.input_catalog_path,
            "margin_threshold": self.margin_threshold,
            "margin_order": self.margin_order,
            "debug_filter_pixel_list": self.debug_filter_pixel_list,
        }
