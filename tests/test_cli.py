"""Tests for cfd_io.cli (typer CLI commands)."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from cfd_io.cli import app
from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers.fortran_binary_direct import read_header
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.hdf5 import write_hdf5

runner = CliRunner()

# -- shared test data --------------------------------------------------
NX, NY, NZ = 5, 4, 2
np.random.seed(55)
GRID = {"x": np.random.rand(NX, NY, NZ), "y": np.random.rand(NX, NY, NZ)}
FLOW = {"uvel": np.random.rand(NX, NY, NZ), "temp": np.random.rand(NX, NY, NZ)}


def _ds(grid, flow=None, attrs=None):
    """Build a Dataset from plain dicts (test helper)."""
    x = grid["x"]
    y = grid["y"]
    z = grid.get("z", np.zeros_like(x))
    return Dataset(
        grid=StructuredGrid(x, y, z),
        flow={k: Field(v) for k, v in (flow or {}).items()},
        attrs=attrs or {},
    )


# ======================================================================
# info subcommand
# ======================================================================

def test_cli_info_hdf5(tmp_path: Path) -> None:
    """CLI 'info' on an HDF5 file prints metadata."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    result = runner.invoke(app, ["info", str(h5)])
    assert result.exit_code == 0
    assert "hdf5" in result.output
    assert str(NX) in result.output


def test_cli_info_hdf5_with_attrs(tmp_path: Path) -> None:
    """CLI 'info --attrs' prints HDF5 attributes when available."""
    h5 = tmp_path / "data_attrs.h5"
    write_hdf5(h5, _ds(GRID, FLOW, attrs={"mach": 6.0, "re1": 1.0e7}))

    result = runner.invoke(app, ["info", str(h5), "--attrs"])
    assert result.exit_code == 0
    assert "attributes:" in result.output
    assert "mach" in result.output
    assert "re1" in result.output


def test_cli_info_split(tmp_path: Path) -> None:
    """CLI 'info' on a .cd file prints split metadata."""
    fpath = tmp_path / "flow.s8"
    gpath = tmp_path / "grid.s8"
    write_binary_direct(fpath, gpath, _ds(GRID, FLOW))

    result = runner.invoke(app, ["info", str(fpath.with_suffix(".cd"))])
    assert result.exit_code == 0
    assert "split" in result.output
    assert "uvel" in result.output


# ======================================================================
# convert subcommand
# ======================================================================

def test_cli_convert_split_to_hdf5(tmp_path: Path) -> None:
    """CLI 'convert' from split to HDF5."""
    fpath = tmp_path / "flow.s8"
    gpath = tmp_path / "grid.s8"
    write_binary_direct(fpath, gpath, _ds(GRID, FLOW))

    out_h5 = tmp_path / "output.h5"
    result = runner.invoke(app, [
        "convert", str(fpath),
        "-o", str(out_h5),
        "-g", str(gpath),
    ])
    assert result.exit_code == 0
    assert out_h5.exists()
    assert "wrote" in result.output


def test_cli_convert_hdf5_to_split(tmp_path: Path) -> None:
    """CLI 'convert' from HDF5 to split."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    out_flow = tmp_path / "out.s8"
    out_grid = tmp_path / "out_grid.s8"
    result = runner.invoke(app, [
        "convert", str(h5),
        "-o", str(out_flow),
        "--grid-out", str(out_grid),
    ])
    assert result.exit_code == 0
    assert out_flow.exists()


def test_cli_convert_hdf5_to_split_swap_ij(tmp_path: Path) -> None:
    """CLI 'convert' from HDF5 to split with --swap-ij."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    out_flow = tmp_path / "out_swap.s8"
    out_grid = tmp_path / "out_swap_grid.s8"
    result = runner.invoke(app, [
        "convert", str(h5),
        "-o", str(out_flow),
        "--grid-out", str(out_grid),
        "--swap-ij",
    ])
    assert result.exit_code == 0

    # verify split header reflects swapped dimensions
    header = read_header(out_flow.with_suffix(".cd"))
    assert header.nx == NY
    assert header.ny == NX
    assert header.nz == NZ


def test_cli_convert_with_attrs(tmp_path: Path) -> None:
    """CLI 'convert' with --mach, --re, --temp-inf flags."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    out = tmp_path / "out.h5"
    result = runner.invoke(app, [
        "convert", str(h5),
        "-o", str(out),
        "--attr", "mach=6.0",
        "--attr", "re=1e7",
        "--attr", "temp_inf=220.0",
    ])
    assert result.exit_code == 0
    assert out.exists()


def test_cli_convert_with_debug(tmp_path: Path) -> None:
    """CLI 'convert' with --debug flag."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    out = tmp_path / "out.h5"
    result = runner.invoke(app, [
        "convert", str(h5),
        "-o", str(out),
        "--debug",
    ])
    assert result.exit_code == 0


# ======================================================================
# no-args shows help
# ======================================================================

def test_cli_no_args() -> None:
    """Invoking the CLI with no args shows help (exit code 0 or 2)."""
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output or "cfd-io" in result.output
