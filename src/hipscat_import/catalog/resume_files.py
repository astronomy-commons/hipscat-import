"""Methods for reading and writing intermediate files for import execution."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hipscat import pixel_math
from hipscat.io import FilePointer, file_io
from numpy import frombuffer


def read_histogram(tmp_path: FilePointer, highest_healpix_order):
    """Read a numpy array at the indicated directory.
    Otherwise, return histogram of appropriate shape."""
    file_name = file_io.append_paths_to_pointer(tmp_path, "mapping_histogram.binary")
    if file_io.does_file_or_directory_exist(file_name):
        with open(file_name, "rb") as file_handle:
            return frombuffer(file_handle.read(), dtype=np.int64)
    return pixel_math.empty_histogram(highest_healpix_order)


def write_histogram(tmp_path: FilePointer, raw_histogram):
    """overwrite existing raw histogram with updated values."""
    file_name = file_io.append_paths_to_pointer(tmp_path, "mapping_histogram.binary")
    with open(file_name, "wb+") as file_handle:
        file_handle.write(raw_histogram.data)


def is_mapping_done(tmp_path: FilePointer):
    """Check for existence of done file"""
    return file_io.does_file_or_directory_exist(
        file_io.append_paths_to_pointer(tmp_path, "mapping_done")
    )


def set_mapping_done(tmp_path: FilePointer):
    """Touch (create) a done file"""
    Path(file_io.append_paths_to_pointer(tmp_path, "mapping_done")).touch()


def is_splitting_done(tmp_path: FilePointer):
    """Check for existence of done file"""
    return file_io.does_file_or_directory_exist(
        file_io.append_paths_to_pointer(tmp_path, "splitting_done")
    )


def set_splitting_done(tmp_path: FilePointer):
    """Touch (create) a done file"""
    Path(file_io.append_paths_to_pointer(tmp_path, "splitting_done")).touch()


def is_reducing_done(tmp_path: FilePointer):
    """Check for existence of done file"""
    return file_io.does_file_or_directory_exist(
        file_io.append_paths_to_pointer(tmp_path, "reducing_done")
    )


def set_reducing_done(tmp_path: FilePointer):
    """Touch (create) a done file"""
    Path(file_io.append_paths_to_pointer(tmp_path, "reducing_done")).touch()


def read_mapping_keys(tmp_path: FilePointer):
    """Read keys from mapping log file"""
    mapping_start_keys = _read_log_keys(
        file_io.append_paths_to_pointer(tmp_path, "mapping_start_log.txt")
    )
    mapping_done_keys = _read_log_keys(
        file_io.append_paths_to_pointer(tmp_path, "mapping_done_log.txt")
    )
    if len(mapping_done_keys) != len(mapping_start_keys):
        raise ValueError(
            "Resume logs are corrupted. Delete temp directory and restart import pipeline."
        )
    return mapping_start_keys


def read_splitting_keys(tmp_path: FilePointer):
    """Read keys from splitting log file"""
    return _read_log_keys(
        file_io.append_paths_to_pointer(tmp_path, "splitting_done_log.txt")
    )


def read_reducing_keys(tmp_path: FilePointer):
    """Read keys from reducing log file"""
    return _read_log_keys(file_io.append_paths_to_pointer(tmp_path, "reducing_log.txt"))


def _read_log_keys(file_name):
    """Read a resume log file, containing timestamp and keys."""
    if file_io.does_file_or_directory_exist(file_name):
        mapping_log = pd.read_csv(
            file_name,
            delimiter="\t",
            header=None,
            names=["time", "key"],
        )
        return mapping_log["key"].tolist()
    return []


def write_mapping_start_key(tmp_path: FilePointer, key):
    """Append single key to mapping log file"""
    _write_log_key(
        file_io.append_paths_to_pointer(tmp_path, "mapping_start_log.txt"), key
    )


def write_mapping_done_key(tmp_path: FilePointer, key):
    """Append single key to mapping log file"""
    _write_log_key(
        file_io.append_paths_to_pointer(tmp_path, "mapping_done_log.txt"), key
    )


def write_splitting_done_key(tmp_path: FilePointer, key):
    """Append single key to splitting log file"""
    _write_log_key(
        file_io.append_paths_to_pointer(tmp_path, "splitting_done_log.txt"), key
    )


def write_reducing_key(tmp_path: FilePointer, key):
    """Append single key to reducing log file"""
    _write_log_key(file_io.append_paths_to_pointer(tmp_path, "reducing_log.txt"), key)


def _write_log_key(file_name, key):
    """Append a tab-delimited line to the file with the current timestamp and provided key"""
    with open(file_name, "a", encoding="utf-8") as mapping_log:
        mapping_log.write(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\t{key}\n')


def clean_resume_files(tmp_path: FilePointer):
    """Remove all intermediate files created in execution."""
    file_io.remove_directory(tmp_path, ignore_errors=True)
