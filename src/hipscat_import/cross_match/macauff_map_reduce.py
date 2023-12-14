import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hipscat.io import file_io, paths
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from hipscat_import.catalog.map_reduce import _get_pixel_directory, _iterate_input_file
from hipscat_import.cross_match.macauff_resume_plan import MacauffResumePlan

# pylint: disable=too-many-arguments,too-many-locals


def split_associations(
    input_file,
    file_reader,
    splitting_key,
    highest_left_order,
    highest_right_order,
    left_alignment,
    right_alignment,
    left_ra_column,
    left_dec_column,
    right_ra_column,
    right_dec_column,
    tmp_path,
):
    """Map a file of links to their healpix pixels and split into shards.


    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """
    for chunk_number, data, mapped_left_pixels in _iterate_input_file(
        input_file, file_reader, highest_left_order, left_ra_column, left_dec_column, False
    ):
        aligned_left_pixels = left_alignment[mapped_left_pixels]
        unique_pixels, unique_inverse = np.unique(aligned_left_pixels, return_inverse=True)

        mapped_right_pixels = hp.ang2pix(
            2**highest_right_order,
            data[right_ra_column].values,
            data[right_dec_column].values,
            lonlat=True,
            nest=True,
        )
        aligned_right_pixels = right_alignment[mapped_right_pixels]

        data["Norder"] = [pix.order for pix in aligned_left_pixels]
        data["Dir"] = [pix.dir for pix in aligned_left_pixels]
        data["Npix"] = [pix.pixel for pix in aligned_left_pixels]

        data["join_Norder"] = [pix.order for pix in aligned_right_pixels]
        data["join_Dir"] = [pix.dir for pix in aligned_right_pixels]
        data["join_Npix"] = [pix.pixel for pix in aligned_right_pixels]

        for unique_index, pixel in enumerate(unique_pixels):
            mapped_indexes = np.where(unique_inverse == unique_index)
            data_indexes = data.index[mapped_indexes[0].tolist()]

            filtered_data = data.filter(items=data_indexes, axis=0)

            pixel_dir = _get_pixel_directory(tmp_path, pixel.order, pixel.pixel)
            file_io.make_directory(pixel_dir, exist_ok=True)
            output_file = file_io.append_paths_to_pointer(
                pixel_dir, f"shard_{splitting_key}_{chunk_number}.parquet"
            )
            filtered_data.to_parquet(output_file, index=False)
        del data, filtered_data, data_indexes

    MacauffResumePlan.splitting_key_done(tmp_path=tmp_path, splitting_key=splitting_key)


def reduce_associations(left_pixel, tmp_path, catalog_path, reduce_key):
    """For all points determined to be in the target left_pixel, map them to the appropriate right_pixel
    and aggregate into a single parquet file."""
    inputs = _get_pixel_directory(tmp_path, left_pixel.order, left_pixel.pixel)

    if not file_io.directory_has_contents(inputs):
        MacauffResumePlan.reducing_key_done(
            tmp_path=tmp_path, reducing_key=f"{left_pixel.order}_{left_pixel.pixel}"
        )
        print(f"Warning: no input data for pixel {left_pixel}")
        return
    destination_dir = paths.pixel_directory(catalog_path, left_pixel.order, left_pixel.pixel)
    file_io.make_directory(destination_dir, exist_ok=True)

    destination_file = paths.pixel_catalog_file(catalog_path, left_pixel.order, left_pixel.pixel)

    merged_table = pq.read_table(inputs)
    dataframe = merged_table.to_pandas().reset_index()

    ## One row group per join_Norder/join_Npix

    join_pixel_frames = dataframe.groupby(["join_Norder", "join_Npix"], group_keys=True)
    join_pixels = [HealpixPixel(pixel[0], pixel[1]) for pixel, _ in join_pixel_frames]
    pixel_argsort = get_pixel_argsort(join_pixels)
    with pq.ParquetWriter(destination_file, merged_table.schema) as writer:
        for pixel_index in pixel_argsort:
            join_pixel = join_pixels[pixel_index]
            join_pixel_frame = join_pixel_frames.get_group((join_pixel.order, join_pixel.pixel)).reset_index()
            writer.write_table(pa.Table.from_pandas(join_pixel_frame, schema=merged_table.schema))

    MacauffResumePlan.reducing_key_done(tmp_path=tmp_path, reducing_key=reduce_key)
