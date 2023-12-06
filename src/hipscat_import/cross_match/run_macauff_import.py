from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat.catalog import Catalog
from hipscat_import.catalog import ResumePlan
from tqdm import tqdm
import hipscat_import.catalog.map_reduce as mr
from hipscat_import.pipeline_resume_plan import PipelineResumePlan
from hipscat import pixel_math
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder
import numpy as np
from hipscat.pixel_tree import PixelAlignment, align_trees
from hipscat.pixel_tree.pixel_alignment_types import PixelAlignmentType
import  healpy as hp

# pylint: disable=unused-argument
def _map_pixels(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if args.resume_plan.is_mapping_done():
        return

    reader_future = client.scatter(args.file_reader)
    futures = []
    for key, file_path in args.resume_plan.map_files:
        futures.append(
            client.submit(
                mr.map_to_pixels,
                key=key,
                input_file=file_path,
                resume_path=args.resume_plan.tmp_path,
                file_reader=reader_future,
                mapping_key=key,
                highest_order=args.mapping_healpix_order,
                ra_column=args.left_ra_column,
                dec_column=args.left_dec_column,
                use_hipscat_index=False,
            )
        )
    args.resume_plan.wait_for_mapping(futures)

def _split_pixels(args, alignment_future, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if args.resume_plan.is_splitting_done():
        return

    reader_future = client.scatter(args.file_reader)
    futures = []
    for key, file_path in args.resume_plan.split_keys:
        futures.append(
            client.submit(
                mr.split_pixels,
                key=key,
                input_file=file_path,
                file_reader=reader_future,
                highest_order=args.mapping_healpix_order,
                ra_column=args.left_ra_column,
                dec_column=args.left_dec_column,
                splitting_key=key,
                cache_shard_path=args.tmp_path,
                resume_path=args.resume_plan.tmp_path,
                alignment=alignment_future,
                use_hipscat_index=False,
            )
        )

    args.resume_plan.wait_for_splitting(futures)

def _reduce_pixels(args, destination_pixel_map, client):
    """Loop over destination pixels and merge into parquet files"""

    if args.resume_plan.is_reducing_done():
        return

    futures = []
    for (
        destination_pixel,
        source_pixels,
        destination_pixel_key,
    ) in args.resume_plan.get_reduce_items(destination_pixel_map):
        futures.append(
            client.submit(
                mr.reduce_pixel_shards,
                key=destination_pixel_key,
                cache_shard_path=args.tmp_path,
                resume_path=args.resume_plan.tmp_path,
                reducing_key=destination_pixel_key,
                destination_pixel_order=destination_pixel.order,
                destination_pixel_number=destination_pixel.pixel,
                destination_pixel_size=source_pixels[0],
                output_path=args.catalog_path,
                ra_column=args.left_ra_column,
                dec_column=args.left_dec_column,
                id_column=args.left_id_column,
                add_hipscat_index=False,
                use_schema_file=False,
                use_hipscat_index=False,
            )
        )

    args.resume_plan.wait_for_reducing(futures)

def run(args, client):
    """run macauff cross-match import pipeline"""
    if not args:
        raise TypeError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise TypeError("args must be type MacauffArguments")

    # Basic plan for map reduce
    # co partitioning the pixels from the left catalog

    # MAP
    #   Get the partition info from left catalog
    # this might be better to read in from `MacauffArguments`
    left_catalog = Catalog.read_from_hipscat(args.left_catalog_dir)
    left_pixel_tree = left_catalog.pixel_tree
    left_pixels = left_catalog.partition_info.get_healpix_pixels()

    #   assign a constant healpix order value for the data
    _map_pixels(args, client)

    with tqdm(
        total=2, desc=PipelineResumePlan.get_formatted_stage_name("Binning"), disable=False
    ) as step_progress:
        raw_histogram = args.resume_plan.read_histogram(args.constant_healpix_order)
        step_progress.update(1)

        # alignment = np.full(len(raw_histogram), None)
        # for pixel_num, pixel_sum in enumerate(raw_histogram):
        #     alignment[pixel_num] = (
        #         args.constant_healpix_order,
        #         pixel_num,
        #         pixel_sum,
        #     )
        nonzero_pix = list(np.nonzero(raw_histogram)[0])
        nonzero_pixels = list(map(
            lambda p : pixel_math.HealpixPixel(args.constant_healpix_order, p),
            nonzero_pix
        ))
        cross_match_pixel_tree = PixelTreeBuilder.from_healpix(healpix_pixels=nonzero_pixels)

        step_progress.update(1)

    alignment = align_trees(
        left=left_pixel_tree,
        right=cross_match_pixel_tree,
        alignment_type=PixelAlignmentType.INNER
    )

    pixel_map = alignment.pixel_mapping

    # generate alignment
    # TODO: make this a utility function in hipscat
    final_alignment = np.full(hp.order2npix(args.mapping_healpix_order), None)
    for pixel in left_pixels:
        df = alignment.pixel_mapping
        sub_df = df.loc[
            (df[alignment.PRIMARY_ORDER_COLUMN_NAME] == pixel.order) & 
            (df[alignment.PRIMARY_PIXEL_COLUMN_NAME] == pixel.pixel)
        ]
        origin_pixels = np.array(sub_df[alignment.JOIN_PIXEL_COLUMN_NAME].values).astype(int)
        row_sum = np.sum(raw_histogram[origin_pixels])
        if row_sum > 0:
            values = [(pixel.order, pixel.pixel, row_sum) for _ in range(len(origin_pixels))]
            final_alignment[origin_pixels] = values

    # generate destination_pixel_map
    # TODO: make this a utility function in hipscat
    destination_pixel_map = {}
    for pixel in left_pixels:
        df = alignment.pixel_mapping
        sub_df = df.loc[
            (df[alignment.PRIMARY_ORDER_COLUMN_NAME] == pixel.order) & 
            (df[alignment.PRIMARY_PIXEL_COLUMN_NAME] == pixel.pixel)
        ]
        origin_pixels = np.array(sub_df[alignment.JOIN_PIXEL_COLUMN_NAME].values).astype(int)
        row_sum = np.sum(raw_histogram[origin_pixels])
        if row_sum > 0:
            destination_pixel_map[pixel] = (row_sum, origin_pixels)

    alignment_future = client.scatter(final_alignment, broadcast=True)
    _split_pixels(args, alignment_future, client)
    _reduce_pixels(args, destination_pixel_map, client)
