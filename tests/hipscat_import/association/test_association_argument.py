"""Tests of argument validation"""


import pytest

from hipscat_import.association.arguments import AssociationArguments

# pylint: disable=protected-access


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        AssociationArguments()


def test_empty_required(tmp_path, small_sky_object_catalog):
    """All non-runtime arguments are required."""
    ## primary_input_catalog_path is missing
    with pytest.raises(ValueError, match="primary_input_catalog_path"):
        AssociationArguments(
            primary_input_catalog_path=None,  ## empty
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
        )

    with pytest.raises(ValueError, match="primary_id_column"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="",  ## empty
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="primary_join_column"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="",  ## empty
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="join_input_catalog_path"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path="",  ## empty
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="join_id_column"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="",  ## empty
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="join_foreign_key"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="",  ## empty
            output_path=tmp_path,
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="output_path"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path="",  ## empty
            output_catalog_name="small_sky_self_join",
            overwrite=True,
        )

    with pytest.raises(ValueError, match="output_catalog_name"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="",  ## empty
            overwrite=True,
        )


def test_column_names(tmp_path, small_sky_object_catalog):
    """Test validation of column names."""
    with pytest.raises(ValueError, match="primary_id_column"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="primary_id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="bad_columns",  ## empty
            overwrite=True,
        )

    with pytest.raises(ValueError, match="join_id_column"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="primary_id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="bad_columns",  ## empty
            overwrite=True,
        )


def test_compute_partition_size(tmp_path, small_sky_object_catalog):
    """Test validation of compute_partition_size."""
    with pytest.raises(ValueError, match="compute_partition_size"):
        AssociationArguments(
            primary_input_catalog_path=small_sky_object_catalog,
            primary_id_column="id",
            primary_join_column="id",
            join_input_catalog_path=small_sky_object_catalog,
            join_id_column="id",
            join_foreign_key="id",
            output_path=tmp_path,
            output_catalog_name="bad_columns",
            compute_partition_size=10,  ## not a valid join option
            overwrite=True,
        )


def test_all_required_args(tmp_path, small_sky_object_catalog):
    """Required arguments are provided."""
    AssociationArguments(
        primary_input_catalog_path=small_sky_object_catalog,
        primary_id_column="id",
        primary_join_column="id",
        join_input_catalog_path=small_sky_object_catalog,
        join_id_column="id",
        join_foreign_key="id",
        output_path=tmp_path,
        output_catalog_name="small_sky_self_join",
    )


def test_to_catalog_info(small_sky_object_catalog, tmp_path):
    """Verify creation of catalog parameters for association table to be created."""
    args = AssociationArguments(
        primary_input_catalog_path=small_sky_object_catalog,
        primary_id_column="id",
        primary_join_column="id",
        join_input_catalog_path=small_sky_object_catalog,
        join_id_column="id",
        join_foreign_key="id",
        output_path=tmp_path,
        output_catalog_name="small_sky_self_join",
    )
    catalog_info = args.to_catalog_info(total_rows=10)
    assert catalog_info.catalog_name == args.output_catalog_name
    assert catalog_info.total_rows == 10


def test_provenance_info(small_sky_object_catalog, tmp_path):
    """Verify that provenance info includes association-specific fields."""
    args = AssociationArguments(
        primary_input_catalog_path=small_sky_object_catalog,
        primary_id_column="id",
        primary_join_column="id",
        join_input_catalog_path=small_sky_object_catalog,
        join_id_column="id",
        join_foreign_key="id",
        output_path=tmp_path,
        output_catalog_name="small_sky_self_join",
    )

    runtime_args = args.provenance_info()["runtime_args"]
    assert "primary_input_catalog_path" in runtime_args
