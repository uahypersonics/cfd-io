"""Tests for error paths and edge cases in cfd_io dispatchers."""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.convert_mod import do_convert, read_file, write_file

# -- shared test data --------------------------------------------------
NX, NY, NZ = 4, 3, 2
np.random.seed(33)
GRID = {"x": np.random.rand(NX, NY, NZ), "y": np.random.rand(NX, NY, NZ)}
FLOW = {"uvel": np.random.rand(NX, NY, NZ)}


# ======================================================================
# read_file error paths
# ======================================================================

def test_read_unsupported_extension(tmp_path: Path) -> None:
    """read_file raises ValueError for unrecognized extension."""
    bad = tmp_path / "data.csv"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="no reader"):
        read_file(bad)


def test_read_split_missing_grid_file(tmp_path: Path) -> None:
    """read_file raises ValueError when split format has no grid_file."""
    from cfd_io.writers.fortran_binary_direct import write_binary_direct

    fpath = tmp_path / "flow.s8"
    gpath = tmp_path / "grid.s8"
    write_binary_direct(fpath, gpath, GRID, FLOW)

    with pytest.raises(ValueError, match="grid file"):
        read_file(fpath)  # no grid_file kwarg


# ======================================================================
# write_file error paths
# ======================================================================

def test_write_unsupported_extension(tmp_path: Path) -> None:
    """write_file raises ValueError for unrecognized extension."""
    bad = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="no writer"):
        write_file(bad, GRID, FLOW)


def test_write_split_missing_grid_file(tmp_path: Path) -> None:
    """write_file raises ValueError when split format has no grid_file."""
    out = tmp_path / "flow.s8"
    with pytest.raises(ValueError, match="grid file"):
        write_file(out, GRID, FLOW)  # no grid_file kwarg


# ======================================================================
# do_convert error paths (exercise the convert wrapper)
# ======================================================================

def test_convert_bad_input_extension(tmp_path: Path) -> None:
    """do_convert raises ValueError for unrecognized input extension."""
    bad_in = tmp_path / "data.csv"
    bad_in.write_text("dummy")
    out = tmp_path / "out.h5"
    with pytest.raises(ValueError, match="no reader"):
        do_convert(bad_in, out)


def test_convert_bad_output_extension(tmp_path: Path) -> None:
    """do_convert raises ValueError for unrecognized output extension."""
    from cfd_io.writers.hdf5 import write_hdf5

    h5 = tmp_path / "data.h5"
    write_hdf5(h5, GRID, FLOW)
    bad_out = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="no writer"):
        do_convert(h5, bad_out)


# ======================================================================
# _resolve_format edge cases
# ======================================================================

def test_read_file_accepts_string_path(tmp_path: Path) -> None:
    """read_file accepts a string path (coerced to Path internally)."""
    from cfd_io.writers.hdf5 import write_hdf5

    h5 = tmp_path / "data.h5"
    write_hdf5(h5, GRID, FLOW)

    grid, flow, _attrs = read_file(str(h5))  # string, not Path
    assert "x" in grid
    assert "uvel" in flow


def test_write_file_accepts_string_path(tmp_path: Path) -> None:
    """write_file accepts a string path (coerced to Path internally)."""
    out = tmp_path / "out.h5"
    result = write_file(str(out), GRID, FLOW)
    assert Path(result).exists()
