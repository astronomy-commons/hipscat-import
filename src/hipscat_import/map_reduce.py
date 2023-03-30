"""Import a set of non-hipscat files using dask for parallelization"""

import healpy as hp
import numpy as np
import pandas as pd
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io, paths

# pylint: disable=too-many-locals,too-many-arguments


def _get_pixel_directory(cache_path: FilePointer, pixel: np.int64):
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
        cache_path, f"dir_{dir_number}", f"pixel_{pixel}"
    )


def map_to_pixels(
    input_file: FilePointer,
    file_reader,
    shard_suffix,
    highest_order,
    ra_column,
    dec_column,
    cache_path: FilePointer = None,
    filter_function=None,
):
    """Map a file of input objects to their healpix pixels.
    
    Args:
        input_file (FilePointer): file to read for catalog data.
        file_reader (InputReader): instance of input reader that
            specifies arguments necessary for reading from the input file.
        shard_suffix (str): unique counter for this input file, used
            when creating intermediate files
        highest_order (int): healpix order to use when mapping
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declation in the dataframe
        cache_path (FilePointer): where to write intermediate files.
            if None, no intermediate files will be written, but healpix
            stats will be returned
        filter_function (function pointer): method to perform some filtering
            or transformation of the input data
    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """

    # Perform checks on the provided path
    if not file_io.does_file_or_directory_exist(input_file):
        raise FileNotFoundError(f"File not found at path: {input_file}")
    if not file_io.is_regular_file(input_file):
        raise FileNotFoundError(
            f"Directory found at path - requires regular file: {input_file}"
        )
    if not file_reader:
        raise NotImplementedError("No file reader implemented")

    required_columns = [ra_column, dec_column]
    histo = pixel_math.empty_histogram(highest_order)

    for chunk_number, data in enumerate(file_reader.read(input_file)):
        data.reset_index(inplace=True)
        if not all(x in data.columns for x in required_columns):
            raise ValueError(
                f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
            )
        # Set up the data we want (filter and find pixel)
        if filter_function:
            data = filter_function(data)
            data.reset_index(inplace=True)
        mapped_pixels = hp.ang2pix(
            2**highest_order,
            data[ra_column].values,
            data[dec_column].values,
            lonlat=True,
            nest=True,
        )
        mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
        histo[mapped_pixel] += count_at_pixel.astype(np.int64)

        if cache_path:
            for pixel in mapped_pixel:
                data_indexes = np.where(mapped_pixels == pixel)
                filtered_data = data.filter(items=data_indexes[0].tolist(), axis=0)

                pixel_dir = _get_pixel_directory(cache_path, pixel)
                file_io.make_directory(pixel_dir, exist_ok=True)
                output_file = file_io.append_paths_to_pointer(
                    pixel_dir, f"shard_{shard_suffix}_{chunk_number}.parquet"
                )
                filtered_data.to_parquet(output_file)
            del filtered_data, data_indexes

        ## Pesky memory!
        del mapped_pixels, mapped_pixel, count_at_pixel
    return histo


def reduce_pixel_shards(
    cache_path,
    origin_pixel_numbers,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    ra_column,
    dec_column,
    id_column,
    add_hipscat_index=True,
    delete_input_files=True,
):
    """Reduce sharded source pixels into destination pixels.
    
    Args:
        cache_path (str): where to read intermediate files
        origin_pixel_numbers (list[int]): high order pixels, with object
            data written to intermediate directories.
        destination_pixel_order (int): order of the final catalog pixel
        destination_pixel_number (int): pixel number at the above order
        destination_pixel_size (int): expected number of rows to write
            for the catalog's final pixel
        output_path (str): where to write the final catalog pixel data
        id_column (str): column for survey identifier, or other sortable column
        delete_input_files (bool): should we delete the intermediate files
            used as input for this method.

    Raises:
        ValueError: if the number of rows written doesn't equal provided
            `destination_pixel_size`    
    """
    destination_dir = paths.pixel_directory(
        output_path, destination_pixel_order, destination_pixel_number
    )
    file_io.make_directory(destination_dir, exist_ok=True)

    destination_file = paths.pixel_catalog_file(
        output_path, destination_pixel_order, destination_pixel_number
    )

    tables = []
    for pixel in origin_pixel_numbers:
        pixel_dir = _get_pixel_directory(cache_path, pixel)

        for file in file_io.get_directory_contents(pixel_dir):
            tables.append(pd.read_parquet(file, engine="pyarrow"))

    merged_table = pd.concat(tables, ignore_index=True, sort=False)
    if id_column:
        merged_table = merged_table.sort_values(by=id_column)
    if add_hipscat_index:
        merged_table["_hipscat_index"] = pixel_math.compute_hipscat_id(
            merged_table[ra_column].values, merged_table[dec_column].values
        )
        merged_table.set_index("_hipscat_index").sort_index()

    rows_written = len(merged_table)

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )

    merged_table.to_parquet(destination_file)

    del merged_table, tables

    if delete_input_files:
        for pixel in origin_pixel_numbers:
            pixel_dir = _get_pixel_directory(cache_path=cache_path, pixel=pixel)

            file_io.remove_directory(pixel_dir, ignore_errors=True)
