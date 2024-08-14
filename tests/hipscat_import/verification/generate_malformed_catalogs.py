import random
import shutil
from pathlib import Path

import attrs
import pyarrow
import pyarrow.dataset
import pyarrow.parquet

DATA_DIR = Path(__file__).parent.parent.parent.parent / "tests/hipscat_import/data"
VALID_CATALOG_DIR = DATA_DIR / "small_sky_object_catalog"
MALFORMED_CATALOGS_DIR = DATA_DIR / "malformed_catalogs"


def run(
    valid_catalog_dir: Path = VALID_CATALOG_DIR, malformed_catalogs_dir: Path = MALFORMED_CATALOGS_DIR
) -> None:
    """Generate malformed catalogs to be used as test data for verification.
    This only needs to be run once unless/until it is desirable to regenerate the dataset.
    """
    Generate.run(valid_catalog_dir=valid_catalog_dir, malformed_catalogs_dir=malformed_catalogs_dir)


@attrs.define
class ValidBase:
    dataset: pyarrow.dataset.Dataset = attrs.field()
    frag: pyarrow.dataset.FileFragment = attrs.field()
    tbl: pyarrow.Table = attrs.field()
    schema: pyarrow.Schema = attrs.field()
    valid_catalog_dir: Path = attrs.field()
    malformed_catalogs_dir: Path = attrs.field()
    insert_dir: str = attrs.field(factory=str)

    @classmethod
    def from_dirs(cls, valid_catalog_dir: Path, malformed_catalogs_dir: Path) -> "ValidBase":
        valid_ds = pyarrow.dataset.parquet_dataset(valid_catalog_dir / "_metadata")
        valid_frag = next(valid_ds.get_fragments())
        valid_tbl = valid_frag.to_table()
        return cls(
            dataset=valid_ds,
            frag=valid_frag,
            tbl=valid_tbl,
            schema=valid_tbl.schema,
            valid_catalog_dir=valid_catalog_dir,
            malformed_catalogs_dir=malformed_catalogs_dir,
        )

    @property
    def fmeta(self) -> Path:
        return self.malformed_catalogs_dir / self.insert_dir / "_metadata"

    @property
    def fcmeta(self) -> Path:
        return self.malformed_catalogs_dir / self.insert_dir / "_common_metadata"

    @property
    def fdata(self) -> Path:
        frag_key = Path(self.frag.path).relative_to(self.valid_catalog_dir)
        return self.malformed_catalogs_dir / self.insert_dir / frag_key


@attrs.define
class Generate:
    def run(
        self,
        valid_catalog_dir: Path = VALID_CATALOG_DIR,
        malformed_catalogs_dir: Path = MALFORMED_CATALOGS_DIR,
    ) -> None:
        """Generate malformed catalogs to be used as test data for verification.
        This only needs to be run once unless/until it is desirable to regenerate the dataset.
        """
        if malformed_catalogs_dir.is_dir():
            print(f"Output directory exists. Remove it and try again.\n{malformed_catalogs_dir}")
            return
        print(f"Generating malformed catalogs from valid catalog at {valid_catalog_dir}...")

        valid = ValidBase.from_dirs(
            valid_catalog_dir=valid_catalog_dir, malformed_catalogs_dir=malformed_catalogs_dir
        )
        generate = Generate()
        generate.valid_truth(valid)
        generate.bad_schemas(valid)
        generate.no_rowgroup_stats(valid)
        generate.wrong_files_and_rows(valid)

    def malformed(self, valid: ValidBase) -> None:
        """Case: <TEMPLATE>"""
        valid.insert_dir = ""
        self._start_new_catalog(valid)
        self._collect_and_write_metadata(valid)
        print(f"Invalid catalog written to {valid.fmeta.parent}")

    def bad_schemas(self, valid: ValidBase) -> None:
        """Case: Files are altered in a way that affects the schema after _metadata gets written."""
        valid.insert_dir = "bad_schemas"
        self._start_new_catalog(valid)

        # Write new files with the correct schema
        fextra_col = valid.fdata.with_suffix(".extra_column.parquet")
        fmissing_col = valid.fdata.with_suffix(".missing_column.parquet")
        fno_metadata = valid.fdata.with_suffix(".no_metadata.parquet")
        fwrong_types = valid.fdata.with_suffix(".wrong_dtypes.parquet")
        for _fout in [fmissing_col, fextra_col, fwrong_types]:
            pyarrow.parquet.write_table(valid.tbl, _fout)

        # Write a _metadata that is correct except for missing file-level metadata
        self._collect_and_write_metadata(valid, schema=valid.schema.remove_metadata())

        # Overwrite the new files using incorrect schemas.
        # drop the file-level metadata
        pyarrow.parquet.write_table(valid.tbl.replace_schema_metadata(None), fno_metadata)
        # drop a column
        pyarrow.parquet.write_table(valid.tbl.drop_columns("dec_error"), fmissing_col)
        # add an extra column
        extra_col = pyarrow.array(random.sample(range(1000), len(valid.tbl)))
        extra_col_tbl = valid.tbl.add_column(5, pyarrow.field("extra", pyarrow.int64()), extra_col)
        pyarrow.parquet.write_table(extra_col_tbl, fextra_col)
        # change some types
        wrong_dtypes = [
            fld if not fld.name.startswith("ra") else fld.with_type(pyarrow.float16()) for fld in valid.schema
        ]
        wrong_dtypes_schema = pyarrow.schema(wrong_dtypes).with_metadata(valid.schema.metadata)
        pyarrow.parquet.write_table(valid.tbl.cast(wrong_dtypes_schema), fwrong_types)

        # Write a _common_metadata with the wrong dtypes.
        pyarrow.parquet.write_metadata(schema=wrong_dtypes_schema, where=valid.fcmeta)

        # Write a _common_metadata with custom metadata and no hipscat columns.
        # This mimics a schema that could have been passed as 'use_schema_file' upon import.
        fcustom_md = valid.fcmeta.with_suffix(".import")
        hipscat_cols = ["_hipscat_index", "Norder", "Dir", "Npix"]
        import_fields = [fld for fld in valid.schema if not fld.name in hipscat_cols]
        import_schema = pyarrow.schema(import_fields).with_metadata({b"custom_key": b"custom_value"})
        pyarrow.parquet.write_metadata(schema=import_schema, where=fcustom_md)

        print(f"Invalid catalog written to {valid.fmeta.parent}")

    def no_rowgroup_stats(self, valid: ValidBase) -> None:
        """Case: ."""
        valid.insert_dir = "no_rowgroup_stats"
        self._start_new_catalog(valid)
        # drop the row group statistics
        pyarrow.parquet.write_table(valid.tbl, valid.fdata, write_statistics=False)
        self._collect_and_write_metadata(valid)
        print(f"Invalid catalog written to {valid.fmeta.parent}")

    def valid_truth(self, valid: ValidBase) -> None:
        """Case: This is the valid catalog that we start with and will be used as the expected truth during testing."""
        valid.insert_dir = "valid_truth"
        base_dir = valid.fmeta.parent
        base_dir.mkdir(parents=True)

        # write a README pointing to the valid_catalog_dir used to generate malformed datasets
        with open(base_dir / "README", "w") as fout:
            fout.writelines(str(valid.valid_catalog_dir.relative_to(DATA_DIR)))

        print(f"Valid, truth README written to {base_dir}")

    def wrong_files_and_rows(self, valid: ValidBase) -> None:
        """Case: ."""
        valid.insert_dir = "wrong_files_and_rows"
        self._start_new_catalog(valid)

        fmissing_file = valid.fdata.with_suffix(".missing_file.parquet")
        fextra_file = valid.fdata.with_suffix(".extra_file.parquet")
        fextra_rows = valid.fdata.with_suffix(".extra_rows.parquet")

        pyarrow.parquet.write_table(valid.tbl, fmissing_file)
        pyarrow.parquet.write_table(valid.tbl, fextra_rows)
        self._collect_and_write_metadata(valid)

        fmissing_file.unlink()
        pyarrow.parquet.write_table(valid.tbl, fextra_file)
        pyarrow.parquet.write_table(self._tbl_with_extra_rows(valid), fextra_rows)

        print(f"Invalid catalog written to {valid.fmeta.parent}")

    def _tbl_with_extra_rows(self, valid: ValidBase) -> pyarrow.Table:
        """Generate a table with extra rows."""
        # generate new rows
        rng = range(len(valid.tbl))
        nrows, new_rows = 2, {}
        for col in valid.tbl.column_names:
            if col not in ("_hipscat_index", "id"):
                # just take a random sample
                new_rows[col] = valid.tbl.column(col).take(random.sample(rng, nrows))
            else:
                # increment the max value to avoid duplicates
                max_id = valid.tbl.column(col).sort()[-1].as_py()
                new_rows[col] = [i + max_id for i in range(1, nrows + 1)]

        # add the rows to the table
        new_tbl = pyarrow.concat_tables([valid.tbl, pyarrow.Table.from_pydict(new_rows, schema=valid.schema)])
        return new_tbl

    @staticmethod
    def _start_new_catalog(valid: ValidBase, with_ancillaries: bool = False) -> None:
        # Start a new catalog by creating the directory and copying in valid files.
        valid.fdata.parent.mkdir(parents=True)
        shutil.copy(valid.frag.path, valid.fdata)

        root_files = valid.valid_catalog_dir.iterdir()
        if not with_ancillaries:
            root_files = [fin for fin in root_files if fin.name.endswith("metadata")]
        for fin in root_files:
            if fin.is_file():
                shutil.copy(fin, valid.malformed_catalogs_dir / valid.insert_dir / fin.name)

    @staticmethod
    def _collect_and_write_metadata(valid: ValidBase, schema: pyarrow.Schema | None = None) -> None:
        base_dir = valid.fmeta.parent
        schema = schema or valid.schema
        ignore_prefixes = [
            ".",
            "_",
            "catalog_info.json",
            "partition_info.csv",
            "point_map.fits",
            "provenance_info.json",
        ]
        dataset = pyarrow.dataset.dataset(base_dir, ignore_prefixes=ignore_prefixes)
        metadata_collector = []
        for frag in dataset.get_fragments():
            frag.ensure_complete_metadata()
            frag.metadata.set_file_path(str(Path(frag.path).relative_to(base_dir)))
            metadata_collector.append(frag.metadata)
        pyarrow.parquet.write_metadata(
            schema=schema, where=valid.fmeta, metadata_collector=metadata_collector
        )
