"""Convert a hipscatted catalog into a HATS catalog, with appropriate metadata/properties."""

import json
import tempfile

import hipscat.pixel_math.healpix_shim as hp
import numpy as np
import pyarrow.parquet as pq
from dask.distributed import as_completed, get_worker
from dask.distributed import print as dask_print
from hats.catalog import CatalogType, PartitionInfo, TableProperties
from hats.io import file_io, parquet_metadata, paths

import hats_import
from hats_import.hipscat_conversion.arguments import ConversionArguments
from hats_import.pipeline_resume_plan import print_progress


def run(args, client):
    """Run index creation pipeline."""
    if not args:
        raise TypeError("args is required and should be type ConversionArguments")
    if not isinstance(args, ConversionArguments):
        raise TypeError("args must be type ConversionArguments")

    # Create basic properties, using catalog info, provenance info, and partition_info files
    catalog_info = None
    with (args.input_catalog_path / "catalog_info.json").open("r", encoding="utf-8") as json_file:
        catalog_info = json.load(json_file)
    provenance_info = None
    with (args.input_catalog_path / "provenance_info.json").open("r", encoding="utf-8") as json_file:
        provenance_info = json.load(json_file)

    catalog_type = CatalogType(catalog_info["catalog_type"])
    if catalog_type not in (
        CatalogType.OBJECT,
        CatalogType.SOURCE,
        CatalogType.MARGIN,
        CatalogType.ASSOCIATION,
    ):
        raise ValueError("Conversion only implemented for object, source, margin, and association tables")

    catalog_info.pop("epoch", None)
    if "tool_args" in provenance_info:
        builder_str = (
            provenance_info["tool_args"]["tool_name"]
            + " v"
            + provenance_info["tool_args"]["version"]
            + " hats-importer conversion v"
            + hats_import.__version__
        )
        catalog_info["hats_builder"] = builder_str
        if "runtime_args" in provenance_info["tool_args"]:
            runtime_args = provenance_info["tool_args"]["runtime_args"]
            catalog_info["hats_cols_sort"] = runtime_args.get("sort_columns", None)
            catalog_info["hats_cols_survey_id"] = runtime_args.get("sort_columns", None)
            catalog_info["hats_max_rows"] = runtime_args.get("pixel_threshold", None)

    partition_info = PartitionInfo.read_from_dir(args.input_catalog_path)
    catalog_info["hats_order"] = partition_info.get_highest_order()

    properties = TableProperties(**catalog_info)

    schema = file_io.read_parquet_metadata(
        args.input_catalog_path / "_common_metadata"
    ).schema.to_arrow_schema()

    futures = []
    for pixel in partition_info.get_healpix_pixels():
        futures.append(
            client.submit(
                _convert_partition_file, pixel, args, schema, properties.ra_column, properties.dec_column
            )
        )
    for future in print_progress(
        as_completed(futures),
        stage_name="Converting Parquet",
        total=len(futures),
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
    ):
        if future.status == "error":
            raise future.exception()

    with print_progress(
        total=4,
        stage_name="Finishing",
        use_progress_bar=args.progress_bar,
        simple_progress_bar=args.simple_progress_bar,
    ) as step_progress:
        total_rows = parquet_metadata.write_parquet_metadata(args.catalog_path)
        assert total_rows == properties.total_rows
        step_progress.update(1)
        file_io.remove_directory(args.tmp_path, ignore_errors=True)
        step_progress.update(1)
        properties.to_properties_file(args.catalog_path)
        partition_info.write_to_file(args.catalog_path / "partition_info.csv")
        step_progress.update(1)
        _write_nested_fits_map(args.input_catalog_path, args.catalog_path)
        step_progress.update(1)


def _convert_partition_file(pixel, args, schema, ra_column, dec_column):
    try:
        input_file = paths.pixel_catalog_file(args.input_catalog_path, pixel)

        table = pq.read_table(input_file, schema=schema)
        num_rows = len(table)

        table = (
            table.drop_columns(["_hipscat_index", "Norder", "Dir", "Npix"])
            .add_column(
                0,
                "_healpix_29",
                [
                    hp.ang2pix(
                        2**29,
                        table[ra_column].to_numpy(),
                        table[dec_column].to_numpy(),
                        nest=True,
                        lonlat=True,
                    )
                ],
            )
            .append_column("Norder", [np.full(num_rows, fill_value=pixel.order, dtype=np.int8)])
            .append_column("Dir", [np.full(num_rows, fill_value=pixel.dir, dtype=np.int64)])
            .append_column("Npix", [np.full(num_rows, fill_value=pixel.pixel, dtype=np.int64)])
        )

        destination_file = paths.pixel_catalog_file(args.catalog_path, pixel)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_file.path, filesystem=destination_file.fs)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        try:
            dask_print("  worker address:", get_worker().address)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        dask_print(exception)


def _write_nested_fits_map(input_dir, output_dir):
    input_file = input_dir / "point_map.fits"
    if not input_file.exists():
        return
    with tempfile.NamedTemporaryFile() as _tmp_file:
        with input_file.open("rb") as _map_file:
            map_data = _map_file.read()
            _tmp_file.write(map_data)
            map_fits_image = hp.read_map(_tmp_file.name, nest=True, h=True)
            header_dict = dict(map_fits_image[1])
            if header_dict["ORDERING"] != "NESTED":
                map_fits_image = hp.read_map(_tmp_file.name)
            else:
                map_fits_image = map_fits_image[0]

    output_file = output_dir / "point_map.fits"
    with tempfile.NamedTemporaryFile() as _tmp_file:
        with output_file.open("wb") as _map_file:
            hp.write_map(
                _tmp_file.name, map_fits_image, overwrite=True, dtype=np.int32, nest=True, coord="CEL"
            )
            _map_file.write(_tmp_file.read())
