"""Tests of macauff arguments"""


from os import path

import pytest

from hipscat_import.cross_match.macauff_arguments import MacauffArguments

# pylint: disable=unused-variable
# pylint: disable=duplicate-code


def test_macauff_arguments(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    """Test that we can create a MacauffArguments instance with two valid catalogs."""
    args = MacauffArguments(
        output_path=tmp_path,
        output_catalog_name="object_to_source",
        tmp_dir=tmp_path,
        left_catalog_dir=small_sky_object_catalog,
        left_ra_column="ra",
        left_dec_column="dec",
        left_id_column="id",
        right_catalog_dir=small_sky_source_catalog,
        right_ra_column="source_ra",
        right_dec_column="source_dec",
        right_id_column="source_id",
        input_path=small_sky_dir,
        input_format="csv",
        metadata_file_path=formats_yaml,
    )

    assert len(args.input_paths) > 0

def test_macauff_arguments_file_list(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    """Test that we can create a MacauffArguments instance with two valid catalogs."""
    files = [path.join(small_sky_dir, "catalog.csv")]
    args = MacauffArguments(
        output_path=tmp_path,
        output_catalog_name="object_to_source",
        tmp_dir=tmp_path,
        left_catalog_dir=small_sky_object_catalog,
        left_ra_column="ra",
        left_dec_column="dec",
        left_id_column="id",
        right_catalog_dir=small_sky_source_catalog,
        right_ra_column="source_ra",
        right_dec_column="source_dec",
        right_id_column="source_id",
        input_file_list=files,
        input_format="csv",
        metadata_file_path=formats_yaml,
    )

    assert len(args.input_paths) > 0


def test_macauff_args_no_input_path(
        small_sky_object_catalog,
        small_sky_source_catalog,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="not provided"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_input_format(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="input_format"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_left_catalog(
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="left_catalog_dir"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_left_id(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="left_id_column"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_invalid_catalog(
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="left_catalog_dir"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_dir, # valid path, but not a catalog
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )


def test_macauff_args_no_right_catalog(
        small_sky_object_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="right_catalog_dir"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_right_catalog_id(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="right_id_column"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_right_invalid_catalog(
        small_sky_object_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(ValueError, match="right_catalog_dir"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_dir, # valid directory with files, not a catalog
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_metadata(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        tmp_path
):
    with pytest.raises(ValueError, match="column metadata file"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
        )

def test_macauff_args_invalid_metadata_file(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        tmp_path
):
    with pytest.raises(ValueError, match="column metadata file must"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="csv",
            metadata_file_path="ceci_n_est_pas_un_fichier.xml",
        )

def test_macauff_args_invalid_input_directory(
        small_sky_object_catalog,
        small_sky_source_catalog,
        formats_yaml,
        tmp_path
):
    with pytest.raises(FileNotFoundError, match="input_path not found"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path="ceci_n_est_pas_un_directoire/",
            input_format="csv",
            metadata_file_path=formats_yaml,
        )

def test_macauff_args_no_files(
        small_sky_object_catalog,
        small_sky_source_catalog,
        small_sky_dir,
        formats_yaml,
        tmp_path
):
    with pytest.raises(FileNotFoundError, match="No input files found"):
        args = MacauffArguments(
            output_path=tmp_path,
            output_catalog_name="object_to_source",
            tmp_dir=tmp_path,
            left_catalog_dir=small_sky_object_catalog,
            left_ra_column="ra",
            left_dec_column="dec",
            left_id_column="id",
            right_catalog_dir=small_sky_source_catalog,
            right_ra_column="source_ra",
            right_dec_column="source_dec",
            right_id_column="source_id",
            input_path=small_sky_dir,
            input_format="parquet", # no files of this format will be found
            metadata_file_path=formats_yaml,
        )
