import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hipscat.io import FilePointer, file_io, paths


def _get_pixel_directory(cache_path: FilePointer, order: np.int64, pixel: np.int64):
    """Create a path for intermediate pixel data.

    This will take the form:

        <cache_path>/dir_<directory separator>/pixel_<pixel>

    where the directory separator is calculated using integer division:

        (pixel/10000)*10000

    and exists to mitigate problems on file systems that don't support
    more than 10_000 children nodes.
    """
    dir_number = int(pixel / 10_000) * 10_000
    return file_io.append_paths_to_pointer(
        cache_path, f"order_{order}", f"dir_{dir_number}", f"pixel_{pixel}"
    )


def reduce_associations(args, left_pixel, highest_right_order, regenerated_right_alignment):
    """For all points determined to be in the target left_pixel, map them to the appropriate right_pixel
    and aggregate into a single parquet file."""
    inputs = _get_pixel_directory(args.tmp_path, left_pixel.order, left_pixel.pixel)

    destination_dir = paths.pixel_directory(args.catalog_path, left_pixel.order, left_pixel.pixel)
    file_io.make_directory(destination_dir, exist_ok=True)

    destination_file = paths.pixel_catalog_file(args.catalog_path, left_pixel.order, left_pixel.pixel)

    tables = []
    tables.append(pq.read_table(inputs))

    merged_table = pa.concat_tables(tables)
    dataframe = merged_table.to_pandas()
    rows_written = len(dataframe)

    dataframe["Norder"] = np.full(rows_written, fill_value=left_pixel.order, dtype=np.uint8)
    dataframe["Dir"] = np.full(rows_written, fill_value=left_pixel.dir, dtype=np.uint32)
    dataframe["Npix"] = np.full(rows_written, fill_value=left_pixel.pixel, dtype=np.uint32)

    mapped_pixels = hp.ang2pix(
        2**highest_right_order,
        dataframe[args.right_ra_column].values,
        dataframe[args.right_dec_column].values,
        lonlat=True,
        nest=True,
    )
    aligned_pixels = regenerated_right_alignment[mapped_pixels]

    dataframe["join_Norder"] = [pixel[0] for pixel in aligned_pixels]
    dataframe["join_Dir"] = [int(pixel[1] / 10_000) * 10_000 for pixel in aligned_pixels]
    dataframe["join_Npix"] = [pixel[1] for pixel in aligned_pixels]

    ## TODO - row groups per join_Norder/join_Npix
    dataframe.to_parquet(destination_file)
