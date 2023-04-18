"""Tests of argument validation, in the absense of command line parsing"""


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


def test_to_catalog_parameters(small_sky_object_catalog, tmp_path):
    """Verify creation of catalog parameters for index to be created."""
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
    catalog_parameters = args.to_catalog_parameters()
    assert catalog_parameters.catalog_name == args.output_catalog_name


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
