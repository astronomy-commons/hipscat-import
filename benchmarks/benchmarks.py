import os
from pathlib import Path

import numpy as np
from hipscat.pixel_math.sparse_histogram import SparseHistogram

from hipscat_import.catalog.resume_plan import ResumePlan


class BinningSuite:
    """Suite that generates sparse array histogram files and benchmarks the operations on them."""

    def setup_cache(self):
        root_dir = Path(os.getcwd())
        tmp_dir = root_dir / "intermediate"
        binning_dir = tmp_dir / "histograms"
        binning_dir.mkdir(parents=True, exist_ok=True)
        max_value = 786_432

        num_paths = 2_000
        for m in range(0, num_paths):
            k = (m + 1) * 100
            pixels = np.arange(k, max_value, k)
            counts = np.full(len(pixels), fill_value=k)

            histo = SparseHistogram.make_from_counts(pixels, counts, healpix_order=8)

            histo.to_file(binning_dir / f"map_{m}")
        return (tmp_dir, num_paths)

    def time_read_histogram(self, cache):
        input_paths = [f"foo{i}" for i in range(0, cache[1])]
        plan = ResumePlan(tmp_path=cache[0], progress_bar=False, input_paths=input_paths)

        plan.read_histogram(8)
