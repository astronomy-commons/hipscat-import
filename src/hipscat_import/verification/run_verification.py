"""Run pass/fail tests and generate verification report of existing hipscat table."""

import collections
import dataclasses
import datetime
import re
from pathlib import Path

import hipscat.io.validation
import pandas as pd
import pyarrow.dataset

from hipscat_import.verification.arguments import VerificationArguments


def run(args: VerificationArguments, write_mode: str = "a"):
    """Run verification pipeline."""
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    verifier = Verifier.from_args(args)
    verifier.test_is_valid_catalog()
    verifier.test_schemas()
    verifier.test_num_rows()

    verifier.record_results(mode=write_mode)
    verifier.record_distributions(mode=write_mode)

    return verifier


Result = collections.namedtuple("Result", ["datetime", "passed", "test", "targets", "description"])
"""Verification test result."""


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S %Z")


@dataclasses.dataclass
class Verifier:
    args: VerificationArguments = dataclasses.field()
    """Arguments to use during verification."""
    files_ds: pyarrow.dataset.Dataset = dataclasses.field()
    """Pyarrow dataset, loaded from the actual files on disk."""
    metadata_ds: pyarrow.dataset.Dataset = dataclasses.field()
    """Pyarrow dataset, loaded from the _metadata file."""
    common_ds: pyarrow.dataset.Dataset = dataclasses.field()
    """Pyarrow dataset, loaded from the _common_metadata file."""
    truth_schema: pyarrow.Schema | None = dataclasses.field(default=None)
    """Pyarrow schema to be used as truth. This will be loaded from args.use_schema_file
    if provided. Else the catalog's _common_metadata file will be used."""
    results: list[Result] = dataclasses.field(default_factory=list)
    """List of results, one for each test that has been done."""
    distributions_df: pd.DataFrame | None = dataclasses.field(default=None)

    @classmethod
    def from_args(cls, args) -> "Verifier":
        # load a dataset from the actual files on disk
        files_ds = pyarrow.dataset.dataset(
            args.input_catalog_path,
            ignore_prefixes=[
                ".",
                "_",
                "catalog_info.json",
                "partition_info.csv",
                "point_map.fits",
                "provenance_info.json",
            ],
        )

        # load a dataset from the _metadata file
        metadata_ds = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_metadata")

        # load a dataset from the _common_metadata file
        common_ds = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_common_metadata")

        # load the input schema if provided, else use the _common_metadata schema
        if args.use_schema_file is not None:
            truth_schema = pyarrow.dataset.parquet_dataset(args.use_schema_file).schema
        else:
            truth_schema = common_ds.schema

        return cls(
            args=args,
            files_ds=files_ds,
            metadata_ds=metadata_ds,
            common_ds=common_ds,
            truth_schema=truth_schema,
        )

    @property
    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def test_is_valid_catalog(self) -> bool:
        test = "is valid catalog"
        target = self.args.input_catalog_path
        # [FIXME] How to get the hipscat version?
        description = "Test that this is a valid HiPSCat catalog using hipscat version <VERSION>."
        print(f"\n{description}")

        passed = hipscat.io.validation.is_valid_catalog(target, strict=True)
        self._append_result(test=test, description=description, passed=passed, targets=target.name)
        return passed

    def test_schemas(self, check_file_metadata: bool = True) -> bool:
        test = "schema"
        _inex = "including file metadata" if check_file_metadata else "excluding file metadata"
        description = f"Test that schemas are equal {_inex}."
        print(f"\n{description}")

        if self.args.use_schema_file is not None:
            # an input schema was provided as truth, so we need to test _common_metadata against it
            truth_src = "input"
            targets = f"_common_metadata vs {truth_src}"
            print(f"\t{targets}")
            passed = self.common_ds.schema.equals(self.truth_schema, check_metadata=check_file_metadata)
            self._append_result(passed=passed, test=test, description=description, targets=targets)
        else:
            truth_src = "_common_metadata"
            passed = True  # just need this variable to exist for return

        # test _metadata schema
        targets = f"_metadata vs {truth_src}"
        print(f"\t{targets}")
        _passed = self.common_ds.schema.equals(self.truth_schema, check_metadata=check_file_metadata)
        self._append_result(passed=_passed, test=test, targets=targets, description=description)

        # test schema in file footers
        targets = f"file footers vs {truth_src}"
        print(f"\t{targets}")
        _passed_ = all(
            frag.physical_schema.equals(self.truth_schema, check_metadata=check_file_metadata)
            for frag in self.files_ds.get_fragments()
        )
        self._append_result(passed=_passed_, test=test, targets=targets, description=description)

        return all([passed, _passed, _passed_])

    def test_num_rows(self) -> bool:
        test = "num rows"
        description = "Test that number of rows are equal."
        print(f"\n{description}")

        # get the number of rows in each file, indexed by partition. we treat this as truth.
        files_df = self._load_nrows(self.files_ds)

        # check _metadata
        targets = "_metadata vs file footers"
        print(f"\t{targets}")
        metadata_df = self._load_nrows(self.metadata_ds)
        passed = metadata_df.equals(files_df)
        self._append_result(passed=passed, test=test, targets=targets, description=description)

        # check total number of rows
        if self.args.expected_total_rows is not None:
            targets = "user total vs file footers"
            print(f"\t{targets}")
            _passed = self.args.expected_total_rows == files_df.num_rows.sum()
            self._append_result(passed=_passed, test=test, targets=targets, description=description)

            return all([passed, _passed])
        return passed

    def _load_nrows(self, dataset: pyarrow.dataset.Dataset) -> pd.DataFrame:
        partition_cols = ["Norder", "Dir", "Npix"]
        nrows_df = pd.DataFrame(
            columns=partition_cols + ["num_rows"],
            data=[
                (
                    int(re.search(r"Norder=(\d+)", frag.path).group(1)),
                    int(re.search(r"Dir=(\d+)", frag.path).group(1)),
                    int(re.search(r"Npix=(\d+)", frag.path).group(1)),
                    frag.metadata.num_rows,
                )
                for frag in dataset.get_fragments()
            ],
        )
        nrows_df = nrows_df.set_index(partition_cols).sort_index()
        return nrows_df

    def _append_result(self, *, test: str, targets: str, description: str, passed: bool):
        self.results.append(
            Result(datetime=now(), passed=passed, test=test, targets=targets, description=description)
        )

    def record_results(self, *, mode: str = "a") -> None:
        fout = Path(self.args.output_path) / "verifier_results.csv"
        fout.parent.mkdir(exist_ok=True, parents=True)
        header = False if (mode == "a" and fout.is_file()) else True
        self.results_df.to_csv(fout, mode=mode, header=header, index=False)
        print(f"\nVerifier results written to {fout}")

    def record_distributions(self, *, mode: str = "a") -> None:
        if self.distributions_df is None:
            print("Gathering distributions (min/max) for fields.")
            rowgrp_stats = [rg.statistics for frag in self.files_ds.get_fragments() for rg in frag.row_groups]
            dist = pd.json_normalize(rowgrp_stats)

            min_ = dist[[f"{c}.min" for c in self.truth_schema.names]].min()
            min_ = min_.rename(index={name: name.removesuffix(".min") for name in min_.index})

            max_ = dist[[f"{c}.max" for c in self.truth_schema.names]].max()
            max_ = max_.rename(index={name: name.removesuffix(".max") for name in max_.index})

            self.distributions_df = pd.DataFrame({"minimum": min_, "maximum": max_}).rename_axis(
                index="field"
            )

        fout = Path(self.args.output_path) / "field_distributions.csv"
        fout.parent.mkdir(exist_ok=True, parents=True)
        header = False if (mode == "a" and fout.is_file()) else True
        self.distributions_df.to_csv(fout, mode=mode, header=header, index=True)
        print(f"Distributions written to {fout}")

        return self.distributions_df
