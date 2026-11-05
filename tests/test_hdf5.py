"""Tests for HDF5 root attributes: dict values round-trip, failed writes leave no file."""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.convert_mod import read_file
from cfd_io.dataset import Dataset, Field, StructuredGrid
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


def _cell_field_dataset() -> Dataset:
    """A 2D vertex grid (5 x 4) with a node field and cell fields (4 x 3), wall at j = 0."""
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 0.3, 4), indexing="ij")
    grid = StructuredGrid(x[:, :, None], y[:, :, None], np.zeros((5, 4, 1)))

    # cell values: velocity grows away from the wall (cfd-io finds the wall by it)
    j_cell = np.arange(3, dtype=float)
    uvel = np.broadcast_to((1.0 + j_cell)[None, :, None], (4, 3, 1)).copy()
    pres = np.arange(12, dtype=float).reshape(4, 3, 1)
    flow = {
        "uvel": Field(uvel, association="cell"),
        "vvel": Field(np.zeros((4, 3, 1)), association="cell"),
        "pres": Field(pres, association="cell"),
        "dist": Field(y[:, :, None].copy()),
    }
    return Dataset(grid=grid, flow=flow)


def test_cell_fields_round_trip_as_cell_fields(tmp_path: Path) -> None:
    """Fields with one point fewer per direction than the grid come back as cell fields."""

    # write and read a vertex grid with cell and node fields
    path = tmp_path / "solution.h5"
    dataset = _cell_field_dataset()
    write_hdf5(path, dataset, dtype="f8")
    read_back = read_hdf5(path)

    # check: locations and values
    assert read_back.flow["pres"].association == "cell"
    assert read_back.flow["uvel"].association == "cell"
    assert read_back.flow["dist"].association == "node"
    np.testing.assert_array_equal(read_back.flow["pres"].data, dataset.flow["pres"].data)


def test_orientation_keeps_cell_fields_cell_fields(tmp_path: Path) -> None:
    """read_file reorients a dataset (wall to j = 0); cell fields stay cell fields."""

    # flip j on disk, so the wall is at the last j-line and read_file must flip it back
    dataset = _cell_field_dataset()
    flipped = Dataset(
        grid=StructuredGrid(
            dataset.grid.x[:, ::-1].copy(), dataset.grid.y[:, ::-1].copy(), dataset.grid.z
        ),
        flow={
            name: Field(field.data[:, ::-1].copy(), association=field.association)
            for name, field in dataset.flow.items()
        },
    )
    path = tmp_path / "flipped.h5"
    write_hdf5(path, flipped, dtype="f8")

    read_back = read_file(path)

    # check: the wall is back at j = 0, cell fields kept their location and values
    np.testing.assert_array_equal(read_back.grid.y[:, :, 0], dataset.grid.y[:, :, 0])
    assert read_back.flow["pres"].association == "cell"
    np.testing.assert_array_equal(read_back.flow["pres"].data, dataset.flow["pres"].data)
