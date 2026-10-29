"""Tests for HDF5 root attributes: dict values round-trip, failed writes leave no file."""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.writers.hdf5 import write_hdf5

# -- shared test data --------------------------------------------------
MARKERS = {"structured": {"imin": "symmetry", "imax": "outflow", "jmin": "wall", "jmax": "farfield"}}


def _grid_dataset(attrs: dict) -> Dataset:
    """Small 2D grid-only dataset with the given root attributes."""
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 3), indexing="ij")
    grid = StructuredGrid(x[:, :, None], y[:, :, None], np.zeros((4, 3, 1)))
    return Dataset(grid=grid, attrs=attrs)


def test_dict_attribute_round_trips(tmp_path: Path) -> None:
    """A nested dict (boundary markers) is written and read back as the same dict."""

    # write a grid with markers next to plain attributes
    path = tmp_path / "grid.h5"
    attrs = {"markers": MARKERS, "mach": 6.0, "grid_type": "body-fitted", "skipped": None}
    write_hdf5(path, _grid_dataset(attrs))

    # read
    read_attrs = read_hdf5(path).attrs

    # check: dict restored, plain values unchanged, None not written
    assert read_attrs["markers"] == MARKERS
    assert read_attrs["mach"] == 6.0
    assert read_attrs["grid_type"] == "body-fitted"
    assert "skipped" not in read_attrs


def test_text_starting_with_a_brace_stays_text(tmp_path: Path) -> None:
    """Only valid JSON objects are decoded; other strings come back unchanged."""

    # write
    path = tmp_path / "grid.h5"
    write_hdf5(path, _grid_dataset({"note": "{not json"}))

    # check
    assert read_hdf5(path).attrs["note"] == "{not json"


@pytest.mark.parametrize("bad_value", [{"array": np.arange(3)}, [{"a": 1}, {"b": 2}]])
def test_unstorable_attribute_names_the_key_and_leaves_no_file(
    tmp_path: Path, bad_value: object
) -> None:
    """A value HDF5 (or JSON) can't store raises a TypeError naming it; no partial file remains."""

    # write
    path = tmp_path / "grid.h5"
    with pytest.raises(TypeError, match="'bad'"):
        write_hdf5(path, _grid_dataset({"bad": bad_value}))

    # check
    assert not path.exists()
