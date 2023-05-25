from dataclasses import dataclass
from hipscat_import.runtime_arguments import RuntimeArguments
from hipscat.catalog import Catalog
from hipscat.io import file_io
import numpy as np
import warnings
import healpy as hp

@dataclass
class MarginCacheArguments(RuntimeArguments):
    """Container for margin cache generation arguments
    Attributes:
        input_catalog_path (str): the path to the hipscat-formatted input catalog.
        margin_output_path (str): the path to the margin cache output directory.
        margin_threshold (float): the size of the margin cache boundary,
            given in arcseconds. setting the `margin_threshold` to be greater than
            the resolution of a `margin_order` healpixel will result in a warning,
            as this may lead to data loss.
        margin_order (int): the order of healpixels that will be used to constrain
            the margin data before doing more precise boundary checking. this value
            must be greater than the highest order of healpix partitioning in
            the source catalog. if `margin_order` is left default or set to -1, then
            the `margin_order` will be set dynamically to the highest partition
            order plus 1.
        overwrite (bool): if there is existing data at the `catalog_path`, should
            we overwrite and create a new catalog.
        pixel_threshold (int): maximum number of rows for a single resulting pixel.
            we may combine hierarchically until we near the `pixel_threshold`
        mapping_healpix_order (int): healpix order to use when mapping. will be
            `highest_healpix_order` unless a positive value is provided for
            `constant_healpix_order`.
        debug_stats_only (bool): do not perform a map reduce and don't create a new
            catalog. generate the partition info.
        tmp_path (str): path for storing intermediate files
        `_tmp_dir` (str): base directory provided by the caller for temporary files.
        progress_bar (bool): if true, a tqdm progress bar will be displayed for user
            feedback of map reduce progress.
        dask_tmp (str): directory for dask worker space. this should be local to
            the execution of the pipeline, for speed of reads and writes.
        dask_n_workers (int): number of workers for the dask client
        dask_threads_per_worker (int): number of threads per dask worker.
    """

    margin_threshold: float = 5.0
    margin_order: int = -1

    input_catalog_path: str = ""
    margin_output_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._check_arguments()

    def _check_arguments(self):
        if not file_io.does_file_or_directory_exist(self.input_catalog_path):
                raise FileNotFoundError("input_catalog_path not found on local storage")

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

    

