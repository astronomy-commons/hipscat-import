from hipscat import pixel_math
from tqdm import tqdm

import hipscat_import.catalog.map_reduce as mr

# import hipscat_import.catalog.run_import as ri
from hipscat_import.catalog.file_readers import CsvReader
from hipscat_import.cross_match.macauff_arguments import MacauffArguments
from hipscat_import.pipeline_resume_plan import PipelineResumePlan


def _map_pixels(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    if args.resume_plan.is_mapping_done():
        return
    
    reader = CsvReader(column_names=args.column_names)

    reader_future = client.scatter(reader)
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
            )
        )
    args.resume_plan.wait_for_mapping(futures)


def run(args, client):
    if not args:
        raise ValueError("args is required and should be type MacauffArguments")
    if not isinstance(args, MacauffArguments):
        raise ValueError("args must be type ImportArguments")
    _map_pixels(args, client)

    with tqdm(
        total=2, desc=PipelineResumePlan.get_formatted_stage_name("Binning"), disable=not args.progress_bar
    ) as step_progress:
        raw_histogram = args.resume_plan.read_histogram(args.mapping_healpix_order)
        step_progress.update(1)
        if args.constant_healpix_order >= 0:
            alignment = np.full(len(raw_histogram), None)
            for pixel_num, pixel_sum in enumerate(raw_histogram):
                alignment[pixel_num] = (
                    args.constant_healpix_order,
                    pixel_num,
                    pixel_sum,
                )

            destination_pixel_map = pixel_math.generate_constant_pixel_map(
                histogram=raw_histogram,
                constant_healpix_order=args.constant_healpix_order,
            )
        else:
            alignment = pixel_math.generate_alignment(
                raw_histogram,
                highest_order=args.highest_healpix_order,
                threshold=args.pixel_threshold,
            )
            destination_pixel_map = pixel_math.compute_pixel_map(
                raw_histogram,
                highest_order=args.highest_healpix_order,
                threshold=args.pixel_threshold,
            )
        step_progress.update(1)



