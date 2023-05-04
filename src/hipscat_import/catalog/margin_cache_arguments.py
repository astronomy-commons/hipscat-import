from dataclasses import dataclass
from hipscat_import.runtime_arguments import RuntimeArguments
from hipscat.catalog import Catalog
import numpy as np
import warnings
import healpy as hp

@dataclass
class MarginCacheArguments(RuntimeArguments):
    """Container for margin cache generation arguments"""

    margin_threshold: float = 5.0
    margin_order: int = -1

    input_catalog_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._check_arguments()

    def _check_arguments(self):
        self.catalog = Catalog(self.input_catalog_path)

        partition_stats = self.catalog.get_pixels()
        highest_order = np.max(partition_stats["Norder"].values)
        margin_pixel_k = highest_order + 1
        if self.margin_order > -1:
            if self.margin_order < margin_pixel_k:
                raise ValueError(
                    "margin_order must be of a higher order than the highest order catalog partition pixel."
                )
        else:
            self.margin_order = margin_pixel_k

        margin_pixel_nside = hp.order2nside(self.margin_order)

        if hp.nside2resol(margin_pixel_nside, arcmin=True) * 60. < self.margin_threshold:
            warnings.warn(
                "Warning: margin pixels have a smaller resolution than margin_threshold; this may lead to data loss in the margin cache."
            )

    

