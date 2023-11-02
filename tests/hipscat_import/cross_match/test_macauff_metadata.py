import os
from xml.etree.ElementTree import ParseError

import pytest
from hipscat.io import file_io

from hipscat_import.cross_match.macauff_metadata import from_xml, from_yaml


def test_from_xml(macauff_data_dir, tmp_path):
    """Test XML file reading and parquet metadata generation."""
    xml_input_file = os.path.join(macauff_data_dir, "macauff_gaia_catwise_match.xml")
    output_file = os.path.join(tmp_path, "output.parquet")

    from_xml(xml_input_file, output_file)

    single_metadata = file_io.read_parquet_metadata(output_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 6


def test_from_xml_malformed(tmp_path):
    """Test some invalid XML file inputs."""
    input_file = os.path.join(tmp_path, "input.parquet")
    output_file = os.path.join(tmp_path, "output.parquet")

    ## No "columns" found at all
    with open(input_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("")

    with pytest.raises(ParseError, match="no element found"):
        from_xml(input_file, output_file)

    ## Some columns, too many required fields
    with open(input_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            """<columns>
	<column>
		<name>Gaia_designation</name>
		<name>The Gaia DR3 object ID.</name>
		<units>long</units>
	</column>
    </columns>"""
        )

    with pytest.raises(ValueError, match="too many name XML elements"):
        from_xml(input_file, output_file)

    ## Unhandled types
    with open(input_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            """<columns>
	<column>
		<name>Gaia_designation</name>
		<description>The Gaia DR3 object ID.</description>
		<units>blob</units>
	</column>
    </columns>"""
        )

    with pytest.raises(ValueError, match="unhandled units blob"):
        from_xml(input_file, output_file)

    ## Some empty fields are ok!
    with open(input_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            """<columns>
	<column>
		<name> </name>
		<units>long </units>
	</column>
    </columns>"""
        )

    from_xml(input_file, output_file)


def test_from_yaml(macauff_data_dir, tmp_path):
    """Test YAML file reading and parquet metadata generation."""
    yaml_input_file = os.path.join(macauff_data_dir, "macauff_gaia_catwise_match_and_nonmatches.yaml")

    from_yaml(yaml_input_file, tmp_path)

    output_file = os.path.join(tmp_path, "macauff_GaiaDR3xCatWISE2020_matches.parquet")
    single_metadata = file_io.read_parquet_metadata(output_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 7

    output_file = os.path.join(tmp_path, "macauff_GaiaDR3xCatWISE2020_gaia_nonmatches.parquet")
    single_metadata = file_io.read_parquet_metadata(output_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 4

    output_file = os.path.join(tmp_path, "macauff_GaiaDR3xCatWISE2020_catwise_nonmatches.parquet")
    single_metadata = file_io.read_parquet_metadata(output_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 4
