"""Tests for cfd_io.info_mod (get_info / FileInfo)."""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.info_mod import get_info
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.hdf5 import write_hdf5
from cfd_io.writers.plot3d import write_plot3d
from cfd_io.writers.tecplot_ascii import write_tecplot_ascii

# -- shared test data --------------------------------------------------
NX, NY, NZ = 6, 4, 2
np.random.seed(77)
GRID = {
    "x": np.random.rand(NX, NY, NZ),
    "y": np.random.rand(NX, NY, NZ),
    "z": np.random.rand(NX, NY, NZ),
}
QFLOW = {
    "dens": np.random.rand(NX, NY, NZ).astype(np.float32),
    "xmom": np.random.rand(NX, NY, NZ).astype(np.float32),
    "ymom": np.random.rand(NX, NY, NZ).astype(np.float32),
    "zmom": np.random.rand(NX, NY, NZ).astype(np.float32),
    "energy": np.random.rand(NX, NY, NZ).astype(np.float32),
}
FLOW = {
    "uvel": np.random.rand(NX, NY, NZ),
    "pres": np.random.rand(NX, NY, NZ),
}


# ======================================================================
# split format (.cd / .s8 / .s4)
# ======================================================================

def test_info_cd(tmp_path: Path) -> None:
    """get_info on a .cd file returns split metadata."""
    fpath = tmp_path / "flow.s8"
    gpath = tmp_path / "grid.s8"
    write_binary_direct(fpath, gpath, GRID, FLOW)

    info = get_info(fpath.with_suffix(".cd"))
    assert info.format == "split"
    assert info.nx == NX
    assert info.ny == NY
    assert info.nz == NZ
    assert info.np == 2
    assert info.var_names == ["uvel", "pres"]
    assert info.precision == "float64"


def test_info_s8(tmp_path: Path) -> None:
    """get_info on a .s8 file auto-resolves the .cd companion."""
    fpath = tmp_path / "flow.s8"
    gpath = tmp_path / "grid.s8"
    write_binary_direct(fpath, gpath, GRID, FLOW)

    info = get_info(fpath)
    assert info.format == "split"
    assert info.nx == NX
    assert info.np == 2


def test_info_s4(tmp_path: Path) -> None:
    """get_info on a .s4 file auto-resolves the .cd companion."""
    fpath = tmp_path / "flow.s4"
    gpath = tmp_path / "grid.s4"
    write_binary_direct(fpath, gpath, GRID, FLOW)

    info = get_info(fpath)
    assert info.format == "split"
    assert info.precision == "float32"


def test_info_s8_missing_cd(tmp_path: Path) -> None:
    """get_info raises FileNotFoundError when .cd is missing."""
    fake = tmp_path / "flow.s8"
    fake.write_bytes(b"\x00")  # create the .s8 but no .cd
    with pytest.raises(FileNotFoundError, match=r"no companion \.cd"):
        get_info(fake)


# ======================================================================
# HDF5
# ======================================================================

def test_info_hdf5(tmp_path: Path) -> None:
    """get_info on a .h5 file returns hdf5 metadata."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, GRID, FLOW)

    info = get_info(h5)
    assert info.format == "hdf5"
    assert info.nx == NX
    assert info.ny == NY
    assert info.nz == NZ
    assert info.np == 2
    assert info.nt >= 1
    assert set(info.var_names) == {"uvel", "pres"}


def test_info_hdf5_multi_timestep(tmp_path: Path) -> None:
    """get_info on a multi-timestep HDF5."""
    h5 = tmp_path / "data.h5"
    write_hdf5(h5, GRID, FLOW, attrs={"timesteps": [100, 200]})

    info = get_info(h5)
    assert info.format == "hdf5"
    assert info.nt >= 1


# ======================================================================
# Plot3D grid (.x)
# ======================================================================

def test_info_plot3d_grid(tmp_path: Path) -> None:
    """get_info on a .x file returns plot3d grid metadata."""
    xfile = tmp_path / "grid.x"
    write_plot3d(xfile, GRID)

    info = get_info(xfile)
    assert info.format == "plot3d"
    assert info.nx == NX
    assert info.ny == NY
    assert info.nz == NZ
    assert info.np == 0  # grid only


# ======================================================================
# Plot3D solution (.q) -- uses FortranBinaryWriter to create .q files
# ======================================================================

def _write_q_binary_3d(fpath: Path, grid: dict, flow: dict) -> None:
    """Write a minimal Plot3D .q file (3-D binary, single block)."""
    from cfd_io.writers.fortran_binary_sequential import FortranBinaryWriter

    nx, ny, nz = next(iter(grid.values())).shape
    with FortranBinaryWriter(fpath) as w:
        w.write_ints([nx, ny, nz])
        w.write_reals([0.0, 0.0, 0.0, 0.0])  # fsmach, alpha, re, time
        parts = [flow[n].ravel(order="F") for n in flow]
        w.write_array_real(np.concatenate(parts).astype(np.float32), fortran_order=False)


def test_info_plot3d_flow(tmp_path: Path) -> None:
    """get_info on a .q file returns plot3d_flow metadata."""
    qfile = tmp_path / "flow.q"
    _write_q_binary_3d(qfile, GRID, QFLOW)

    info = get_info(qfile)
    assert info.format == "plot3d_flow"
    assert info.nx == NX
    assert info.ny == NY
    assert info.nz == NZ
    assert info.np == 5
    assert info.nt == 1


# ======================================================================
# Tecplot ASCII (.dat)
# ======================================================================

def test_info_tecplot_ascii(tmp_path: Path) -> None:
    """get_info on a .dat file returns tecplot metadata."""
    dat = tmp_path / "data.dat"
    write_tecplot_ascii(dat, GRID, FLOW)

    info = get_info(dat)
    assert info.format == "tecplot"
    assert info.nx == NX
    assert info.ny == NY
    assert info.nz == NZ
    assert info.np == 2
    assert info.nt == 1


# ======================================================================
# error paths
# ======================================================================

def test_info_unsupported_extension(tmp_path: Path) -> None:
    """get_info raises ValueError for unknown file extension."""
    bad = tmp_path / "file.csv"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="unsupported file type"):
        get_info(bad)
