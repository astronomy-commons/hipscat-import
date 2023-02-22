"""Methods for reading and writing intermediate files for import execution."""

import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hipscat import pixel_math


def read_histogram(tmp_path, highest_healpix_order):
    """Read a numpy array at the indicated directory.
    Otherwise, return histogram of appropriate shape."""
    if os.path.exists(os.path.join(tmp_path, "mapping_histogram.csv")):
        return np.loadtxt(
            os.path.join(tmp_path, "mapping_histogram.csv"),
            dtype=np.ulonglong,
            delimiter=",",
        )
    return pixel_math.empty_histogram(highest_healpix_order)


def write_histogram(tmp_path, raw_histogram):
    """overwrite existing raw histogram with updated values."""
    np.savetxt(
        os.path.join(tmp_path, "mapping_histogram.csv"),
        raw_histogram,
        fmt="%i",
        delimiter=",",
    )


def is_mapping_done(tmp_path):
    """Check for existence of done file"""
    return os.path.exists(os.path.join(tmp_path, "mapping_done"))


def set_mapping_done(tmp_path):
    """Touch (create) a done file"""
    Path(os.path.join(tmp_path, "mapping_done")).touch()


def is_reducing_done(tmp_path):
    """Check for existence of done file"""
    return os.path.exists(os.path.join(tmp_path, "reducing_done"))


def set_reducing_done(tmp_path):
    """Touch (create) a done file"""
    Path(os.path.join(tmp_path, "reducing_done")).touch()


def read_mapping_keys(tmp_path):
    """Read keys from mapping log file"""
    return _read_log_keys(os.path.join(tmp_path, "mapping_log.txt"))


def read_reducing_keys(tmp_path):
    """Read keys from reducing log file"""
    return _read_log_keys(os.path.join(tmp_path, "reducing_log.txt"))


def _read_log_keys(file_name):
    """Read a resume log file, containing timestamp and keys."""
    if os.path.exists(file_name):
        mapping_log = pd.read_csv(
            file_name,
            delimiter="\t",
            header=None,
            names=["time", "key"],
        )
        return mapping_log["key"].values
    return []


def write_mapping_key(tmp_path, key):
    """Append single key to mapping log file"""
    _write_log_key(os.path.join(tmp_path, "mapping_log.txt"), key)


def write_reducing_key(tmp_path, key):
    """Append single key to reducing log file"""
    _write_log_key(os.path.join(tmp_path, "reducing_log.txt"), key)


def _write_log_key(file_name, key):
    """Append a tab-delimited line to the file with the current timestamp and provided key"""
    with open(file_name, "a", encoding="utf-8") as mapping_log:
        mapping_log.write(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\t{key}\n')


def clean_resume_files(tmp_path):
    """Remove all intermediate files created in execution."""
    shutil.rmtree(tmp_path, ignore_errors=True)
