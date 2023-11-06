"""Utilities for generating parquet metdata from macauff-generated metadata files."""

import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hipscat.io import file_io


def _get_inner_xml_value(parent_el, node_type, default_value):
    child_el = parent_el.findall(node_type)
    if len(child_el) == 0:
        return default_value
    if len(child_el) > 1:
        raise ValueError(f"found too many {node_type} XML elements")
    return child_el[0].text.strip()


def _construct_field(name, units, metadata_dict):
    """Helper method to construct a pyarrow field from macauff metadata strings."""
    if units == "string":
        pa_type = pa.string()
    elif units in ("float", "double"):
        pa_type = pa.float64()
    elif units in ("integer", "long"):
        pa_type = pa.int64()
    else:
        raise ValueError(f"unhandled units {units}")
    return pa.field(name, pa_type, metadata=metadata_dict)


def from_xml(input_file, output_file):
    """Read XML file with column metadata for a cross-match file from macauff.

    Expects XML with the format::

        <columns>
            <column>
                <name>$COLUMN_NAME</name>
                <description>$COLUMN_DESCRIPTION</description>
                <units>$COLUMN_UNIT_DESCRIPTOR</units>
            </column>
        </columns>

    Args:
        input file (str): file to read for match metadata
        output_file (str): desired location for output parquet metadata file

    Raises
        ValueError: if the XML is mal-formed
    """
    fields = []
    root_el = ET.parse(input_file).getroot()
    columns = root_el.findall("column")

    for column in columns:
        name = _get_inner_xml_value(column, "name", "foo")
        description = _get_inner_xml_value(column, "description", "")
        units = _get_inner_xml_value(column, "units", "string")

        fields.append(_construct_field(name, units, metadata_dict={"macauff_description": description}))

    schema = pa.schema(fields)
    pq.write_table(schema.empty_table(), where=output_file)


def from_yaml(input_file, output_directory):
    """Read YAML file with column metadata for the various cross-match files from macauff.

    Expects YAML with the format::

        name: macauff_GaiaDR3xCatWISE2020
        description: Match and non-match table for macauff cross-matches of Gaia DR3 and CatWISE 2020.
        tables:
        - name: macauff_GaiaDR3xCatWISE2020_matches
          "@id": "#macauff_GaiaDR3xCatWISE2020_matches"
          description: Counterpart associations between Gaia and WISE
          columns:
          - name: gaia_source_id
            datatype: long
            description: The Gaia DR3 object ID.

    Args:
        input file (str): file to read for match metadata
        output_dir (str): desired location for output parquet metadata files
            We will write one file per table in the "tables" element.
    """
    with open(input_file, "r", encoding="utf-8") as file_handle:
        metadata = yaml.safe_load(file_handle)
        tables = metadata.get("tables", [])
        for index, table in enumerate(tables):
            fields = []
            table_name = table.get("name", f"metadata_table_{index}")
            for col_index, column in enumerate(table.get("columns", [])):
                name = column.get("name", f"column_{col_index}")
                units = column.get("units", "string")
                fields.append(_construct_field(name, units, metadata_dict=column))

            schema = pa.schema(fields)
            output_file = file_io.append_paths_to_pointer(output_directory, f"{table_name}.parquet")
            pq.write_table(schema.empty_table(), where=str(output_file))
