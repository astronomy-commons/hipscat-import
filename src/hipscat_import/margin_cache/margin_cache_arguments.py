import warnings
from dataclasses import dataclass

import healpy as hp
import numpy as np
from hipscat.catalog import Catalog
from hipscat.catalog.margin_cache.margin_cache_catalog_info import (
    MarginCacheCatalogInfo,
)
from hipscat.io import file_io

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

    input_catalog_path: str = ""
    """the path to the hipscat-formatted input catalog."""

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        super()._check_arguments()
        if not file_io.does_file_or_directory_exist(self.input_catalog_path):
            raise FileNotFoundError("input_catalog_path not found on local storage")

        self.catalog = Catalog.read_from_hipscat(self.input_catalog_path)

        partition_stats = self.catalog.get_pixels()
        highest_order = np.max(partition_stats["Norder"].values)
        margin_pixel_k = highest_order + 1
        if self.margin_order > -1:
            if self.margin_order < margin_pixel_k:
                # pylint: disable=line-too-long
                raise ValueError(
                    "margin_order must be of a higher order "
                    "than the highest order catalog partition pixel."
                )
                # pylint: enable=line-too-long
        else:
            self.margin_order = margin_pixel_k

        margin_pixel_nside = hp.order2nside(self.margin_order)

        if (
            hp.nside2resol(margin_pixel_nside, arcmin=True) * 60.0
            < self.margin_threshold
        ):
            # pylint: disable=line-too-long
            warnings.warn(
                "Warning: margin pixels have a smaller resolution than margin_threshold; this may lead to data loss in the margin cache."
            )
            # pylint: enable=line-too-long

    def to_catalog_info(self, total_rows) -> MarginCacheCatalogInfo:
        """Catalog-type-specific dataset info."""
        info = {
            "catalog_name": self.output_catalog_name,
            "total_rows": total_rows,
            "catalog_type": "margin",
            "primary_catalog": self.input_catalog_path,
            "margin_threshold": self.margin_threshold,
        }
        return MarginCacheCatalogInfo(**info)

    def additional_runtime_provenance_info(self) -> dict:
        return {
            "input_catalog_path": str(self.input_catalog_path),
            "margin_threshold": self.margin_threshold,
            "margin_order": self.margin_order,
        }
