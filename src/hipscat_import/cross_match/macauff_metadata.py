"""Utilities for generating parquet metdata from macauff-generated metadata files."""

import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq


def _get_inner_xml_value(parent_el, node_type, default_value):
    child_el = parent_el.findall(node_type)
    if len(child_el) == 0:
        return default_value
    if len(child_el) > 1:
        raise ValueError(f"found too many {node_type} XML elements")
    return child_el[0].text.strip()


def from_xml(input_file, output_file):
    """Read XML file with column metadata for a cross-match file from macauff.
    
    Expects XML with the format:

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
        units = _get_inner_xml_value(column,"units", "string")
        metadata = {"macauff_description": description}

        if units == "string":
            pa_type = pa.string()
        elif units == "float" or units =="double":
            pa_type = pa.float64()
        elif units == "integer" or units == "long":
            pa_type = pa.int64()
        else:
            raise ValueError(f"unhandled units {units}")
        fields.append(pa.field(name, pa_type,metadata=metadata))

    schema = pa.schema(fields)
    pq.write_table(schema.empty_table(), where=output_file)

def from_yaml(input_file, output_file):
    pass
