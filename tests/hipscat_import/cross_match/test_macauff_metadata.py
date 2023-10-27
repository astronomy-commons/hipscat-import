import os

import pyarrow as pa
from hipscat.io import file_io

from hipscat_import.cross_match.macauff_metadata import from_xml


def test_from_xml(macauff_data_dir, tmp_path):
    """Test XML file reading and parquet metadata generation."""
    xml_input_file = os.path.join(macauff_data_dir, "macauff_gaia_catwise_match.xml")
    output_file = os.path.join(tmp_path, "output.parquet")

    from_xml(xml_input_file, output_file)

    single_metadata = file_io.read_parquet_metadata(output_file)
    schema = single_metadata.schema.to_arrow_schema()

    assert len(schema) == 22

def test_from_yaml(macauff_data_dir, tmp_path):
    """Test YAML file reading and parquet metadata generation."""
    pass