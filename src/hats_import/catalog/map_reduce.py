"""Import a set of non-hats files using dask for parallelization"""

import pickle

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hats import pixel_math
from hats.io import file_io, paths
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, spatial_index_to_healpix
from upath import UPath

from hats_import.catalog.resume_plan import ResumePlan
from hats_import.catalog.sparse_histogram import SparseHistogram
from hats_import.pipeline_resume_plan import get_pixel_cache_directory, print_task_failure

# pylint: disable=too-many-locals,too-many-arguments


def _has_named_index(dataframe):
    """Heuristic to determine if a dataframe has some meaningful index.

    This will reject dataframes with no index name for a single index,
    or empty names for multi-index (e.g. [] or [None]).
    """
    if dataframe.index.name is not None:
        ## Single index with a given name.
        return True
    if len(dataframe.index.names) == 0 or all(name is None for name in dataframe.index.names):
        return False
    return True


def _iterate_input_file(
    input_file: UPath,
    pickled_reader_file: str,
    highest_order,
    ra_column,
    dec_column,
    use_healpix_29=False,
    read_columns=None,
):
    """Helper function to handle input file reading and healpix pixel calculation"""
    with open(pickled_reader_file, "rb") as pickle_file:
        file_reader = pickle.load(pickle_file)
    if not file_reader:
        raise NotImplementedError("No file reader implemented")

    for chunk_number, data in enumerate(file_reader.read(input_file, read_columns=read_columns)):
        if use_healpix_29:
            if data.index.name == SPATIAL_INDEX_COLUMN:
                mapped_pixels = spatial_index_to_healpix(data.index, target_order=highest_order)
            else:
                mapped_pixels = spatial_index_to_healpix(
                    data[SPATIAL_INDEX_COLUMN], target_order=highest_order
                )
        else:
            # Set up the pixel data
            mapped_pixels = hp.ang2pix(
                2**highest_order,
                data[ra_column].to_numpy(copy=False, dtype=float),
                data[dec_column].to_numpy(copy=False, dtype=float),
                lonlat=True,
                nest=True,
            )
        yield chunk_number, data, mapped_pixels


def map_to_pixels(
    input_file: UPath,
    pickled_reader_file: str,
    resume_path: UPath,
    mapping_key,
    highest_order,
    ra_column,
    dec_column,
    use_healpix_29=False,
):
    """Map a file of input objects to their healpix pixels.

    Args:
        input_file (UPath): file to read for catalog data.
        file_reader (hats_import.catalog.file_readers.InputReader): instance of input
            reader that specifies arguments necessary for reading from the input file.
        resume_path (UPath): where to write resume partial results.
        mapping_key (str): unique counter for this input file, used
            when creating intermediate files
        highest_order (int): healpix order to use when mapping
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declation in the dataframe

    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """
    try:
        histo = SparseHistogram.make_empty(highest_order)

        if use_healpix_29:
            read_columns = [SPATIAL_INDEX_COLUMN]
        else:
            read_columns = [ra_column, dec_column]

        for _, _, mapped_pixels in _iterate_input_file(
            input_file,
            pickled_reader_file,
            highest_order,
            ra_column,
            dec_column,
            use_healpix_29,
            read_columns,
        ):
            mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)

            partial = SparseHistogram.make_from_counts(
                mapped_pixel, count_at_pixel, healpix_order=highest_order
            )
            histo.add(partial)

        histo.to_file(ResumePlan.partial_histogram_file(tmp_path=resume_path, mapping_key=mapping_key))
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING stage with file {input_file}", exception)
        raise exception


def split_pixels(
    input_file: UPath,
    pickled_reader_file: str,
    splitting_key,
    highest_order,
    ra_column,
    dec_column,
    cache_shard_path: UPath,
    resume_path: UPath,
    alignment_file=None,
    use_healpix_29=False,
):
    """Map a file of input objects to their healpix pixels and split into shards.

    Args:
        input_file (UPath): file to read for catalog data.
        file_reader (hats_import.catalog.file_readers.InputReader): instance
            of input reader that specifies arguments necessary for reading from the input file.
        splitting_key (str): unique counter for this input file, used
            when creating intermediate files
        highest_order (int): healpix order to use when mapping
        ra_column (str): where to find right ascension data in the dataframe
        dec_column (str): where to find declation in the dataframe
        cache_shard_path (UPath): where to write intermediate parquet files.
        resume_path (UPath): where to write resume files.

    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: if the file does not exist, or is a directory
    """
    try:
        with open(alignment_file, "rb") as pickle_file:
            alignment = pickle.load(pickle_file)
        for chunk_number, data, mapped_pixels in _iterate_input_file(
            input_file, pickled_reader_file, highest_order, ra_column, dec_column, use_healpix_29
        ):
            aligned_pixels = alignment[mapped_pixels]
            unique_pixels, unique_inverse = np.unique(aligned_pixels, return_inverse=True)

            for unique_index, [order, pixel, _] in enumerate(unique_pixels):
                filtered_data = data.iloc[unique_inverse == unique_index]

                pixel_dir = get_pixel_cache_directory(cache_shard_path, HealpixPixel(order, pixel))
                file_io.make_directory(pixel_dir, exist_ok=True)
                output_file = file_io.append_paths_to_pointer(
                    pixel_dir, f"shard_{splitting_key}_{chunk_number}.parquet"
                )
                if _has_named_index(filtered_data):
                    filtered_data.to_parquet(output_file.path, index=True, filesystem=output_file.fs)
                else:
                    filtered_data.to_parquet(output_file.path, index=False, filesystem=output_file.fs)
                del filtered_data

        ResumePlan.splitting_key_done(tmp_path=resume_path, splitting_key=splitting_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed SPLITTING stage with file {input_file}", exception)
        raise exception


def reduce_pixel_shards(
    cache_shard_path,
    resume_path,
    reducing_key,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    ra_column,
    dec_column,
    sort_columns: str = "",
    use_healpix_29=False,
    add_healpix_29=True,
    delete_input_files=True,
    use_schema_file="",
):
    """Reduce sharded source pixels into destination pixels.

    In addition to combining multiple shards of data into a single
    parquet file, this method will add a few new columns:

        - ``Norder`` - the healpix order for the pixel
        - ``Dir`` - the directory part, corresponding to the pixel
        - ``Npix`` - the healpix pixel
        - ``_healpix_29`` - optional - a spatially-correlated
          64-bit index field.

    Notes on ``_healpix_29``:

        - if we generate the field, we will promote any previous
          *named* pandas index field(s) to a column with
          that name.
        - see ``hats.pixel_math.spatial_index``
          for more in-depth discussion of this field.

    Args:
        cache_shard_path (UPath): where to read intermediate parquet files.
        resume_path (UPath): where to write resume files.
        reducing_key (str): unique string for this task, used for resume files.
        origin_pixel_numbers (list[int]): high order pixels, with object
            data written to intermediate directories.
        destination_pixel_order (int): order of the final catalog pixel
        destination_pixel_number (int): pixel number at the above order
        destination_pixel_size (int): expected number of rows to write
            for the catalog's final pixel
        output_path (UPath): where to write the final catalog pixel data
        sort_columns (str): column for survey identifier, or other sortable column
        add_healpix_29 (bool): should we add a _healpix_29 column to
            the resulting parquet file?
        delete_input_files (bool): should we delete the intermediate files
            used as input for this method.
        use_schema_file (str): use the parquet schema from the indicated
            parquet file.

    Raises:
        ValueError: if the number of rows written doesn't equal provided
            `destination_pixel_size`
    """
    try:
        destination_dir = paths.pixel_directory(
            output_path, destination_pixel_order, destination_pixel_number
        )
        file_io.make_directory(destination_dir, exist_ok=True)

        healpix_pixel = HealpixPixel(destination_pixel_order, destination_pixel_number)
        destination_file = paths.pixel_catalog_file(output_path, healpix_pixel)

        schema = None
        if use_schema_file:
            schema = file_io.read_parquet_metadata(use_schema_file).schema.to_arrow_schema()

        tables = []
        pixel_dir = get_pixel_cache_directory(cache_shard_path, healpix_pixel)

        if schema:
            tables.append(pq.read_table(pixel_dir, schema=schema))
        else:
            tables.append(pq.read_table(pixel_dir))

        merged_table = pa.concat_tables(tables)

        rows_written = len(merged_table)

        if rows_written != destination_pixel_size:
            raise ValueError(
                "Unexpected number of objects at pixel "
                f"({healpix_pixel})."
                f" Expected {destination_pixel_size}, wrote {rows_written}"
            )

        dataframe = merged_table.to_pandas()
        if sort_columns:
            dataframe = dataframe.sort_values(sort_columns.split(","))
        if add_healpix_29:
            ## If we had a meaningful index before, preserve it as a column.
            if _has_named_index(dataframe):
                dataframe = dataframe.reset_index()

            dataframe[SPATIAL_INDEX_COLUMN] = pixel_math.compute_spatial_index(
                dataframe[ra_column].values,
                dataframe[dec_column].values,
            )
            dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN).sort_index()

            # Adjust the schema to make sure that the _healpix_29 will
            # be saved as a uint64
        elif use_healpix_29:
            if dataframe.index.name != SPATIAL_INDEX_COLUMN:
                dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN)
            dataframe = dataframe.sort_index()

        dataframe["Norder"] = np.full(rows_written, fill_value=healpix_pixel.order, dtype=np.uint8)
        dataframe["Dir"] = np.full(rows_written, fill_value=healpix_pixel.dir, dtype=np.uint64)
        dataframe["Npix"] = np.full(rows_written, fill_value=healpix_pixel.pixel, dtype=np.uint64)

        if schema:
            schema = _modify_arrow_schema(schema, add_healpix_29)
            dataframe.to_parquet(destination_file.path, schema=schema, filesystem=destination_file.fs)
        else:
            dataframe.to_parquet(destination_file.path, filesystem=destination_file.fs)

        del dataframe, merged_table, tables

        if delete_input_files:
            pixel_dir = get_pixel_cache_directory(cache_shard_path, healpix_pixel)

            file_io.remove_directory(pixel_dir, ignore_errors=True)

        ResumePlan.reducing_key_done(tmp_path=resume_path, reducing_key=reducing_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(
            f"Failed REDUCING stage for shard: {destination_pixel_order} {destination_pixel_number}",
            exception,
        )
        raise exception


def _modify_arrow_schema(schema, add_healpix_29):
    if add_healpix_29:
        pandas_index_column = schema.get_field_index("__index_level_0__")
        if pandas_index_column != -1:
            schema = schema.remove(pandas_index_column)
        schema = schema.insert(0, pa.field(SPATIAL_INDEX_COLUMN, pa.int64()))
    schema = (
        schema.append(pa.field("Norder", pa.uint8()))
        .append(pa.field("Dir", pa.uint64()))
        .append(pa.field("Npix", pa.uint64()))
    )

    return schema
