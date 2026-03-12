"""Smoke tests for Plot3D, Tecplot ASCII, and grid-only conversion."""

from pathlib import Path

import numpy as np

from cfd_io.convert_mod import do_convert, read_file
from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers.fortran_binary_direct import read_binary_direct
from cfd_io.readers.plot3d import read_plot3d
from cfd_io.readers.plot3d_flow import read_plot3d_flow
from cfd_io.readers.plot3d_flow_ascii import read_plot3d_flow_ascii
from cfd_io.readers.plot3d_flow_binary import read_plot3d_flow_binary
from cfd_io.readers.tecplot_ascii import read_tecplot_ascii
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.fortran_binary_sequential import FortranBinaryWriter
from cfd_io.writers.plot3d import write_plot3d
from cfd_io.writers.tecplot_ascii import write_tecplot_ascii

# -- shared test data --------------------------------------------------
NX, NY, NZ = 12, 8, 3
np.random.seed(99)
GRID_3D = {
    "x": np.random.rand(NX, NY, NZ),
    "y": np.random.rand(NX, NY, NZ),
    "z": np.random.rand(NX, NY, NZ),
}
GRID_2D = {
    "x": np.random.rand(NX, NY, 1),
    "y": np.random.rand(NX, NY, 1),
}
FLOW = {
    "uvel": np.random.rand(NX, NY, NZ),
    "pres": np.random.rand(NX, NY, NZ),
}


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
# Plot3D tests
# ======================================================================

def test_plot3d_binary_3d_roundtrip(tmp_path: Path) -> None:
    """3-D Plot3D binary round-trip."""
    p = tmp_path / "grid.x"
    write_plot3d(p, _ds(GRID_3D), binary=True)
    ds = read_plot3d(p)

    assert np.allclose(ds.grid.x, GRID_3D["x"])
    assert np.allclose(ds.grid.y, GRID_3D["y"])
    assert np.allclose(ds.grid.z, GRID_3D["z"])
    assert len(ds.flow) == 0


def test_plot3d_ascii_3d_roundtrip(tmp_path: Path) -> None:
    """3-D Plot3D ASCII round-trip."""
    p = tmp_path / "grid.x"
    write_plot3d(p, _ds(GRID_3D), binary=False)
    ds = read_plot3d(p)

    assert np.allclose(ds.grid.x, GRID_3D["x"], atol=1e-10)
    assert np.allclose(ds.grid.z, GRID_3D["z"], atol=1e-10)


def test_plot3d_binary_2d_roundtrip(tmp_path: Path) -> None:
    """2-D Plot3D binary round-trip (no z coordinate)."""
    p = tmp_path / "grid.x"
    write_plot3d(p, _ds(GRID_2D), binary=True)
    ds = read_plot3d(p)

    assert np.allclose(ds.grid.x, GRID_2D["x"])
    assert ds.grid.ndim == 2


def test_plot3d_ascii_2d_roundtrip(tmp_path: Path) -> None:
    """2-D Plot3D ASCII round-trip."""
    p = tmp_path / "grid.x"
    write_plot3d(p, _ds(GRID_2D), binary=False)
    ds = read_plot3d(p)

    assert np.allclose(ds.grid.x, GRID_2D["x"], atol=1e-10)
    assert ds.grid.ndim == 2


# ======================================================================
# Tecplot ASCII tests
# ======================================================================

def test_tecplot_ascii_point_3d_roundtrip(tmp_path: Path) -> None:
    """3-D Tecplot ASCII POINT-format round-trip."""
    p = tmp_path / "data.dat"
    write_tecplot_ascii(p, _ds(GRID_3D, FLOW), title="test3d")
    ds = read_tecplot_ascii(p)

    assert np.allclose(ds.grid.x, GRID_3D["x"], atol=1e-6)
    assert "uvel" in ds.flow and "pres" in ds.flow
    assert ds.attrs.get("title") == "test3d"
    assert np.allclose(ds.flow["uvel"].data, FLOW["uvel"], atol=1e-6)


def test_tecplot_ascii_2d_roundtrip(tmp_path: Path) -> None:
    """2-D Tecplot ASCII round-trip (no z, no flow)."""
    p = tmp_path / "grid.dat"
    write_tecplot_ascii(p, _ds(GRID_2D))
    ds = read_tecplot_ascii(p)

    assert len(ds.flow) == 0
    assert np.allclose(ds.grid.x, GRID_2D["x"], atol=1e-6)


def test_tecplot_ascii_grid_only(tmp_path: Path) -> None:
    """3-D Tecplot ASCII with grid only (no flow variables)."""
    p = tmp_path / "grid.dat"
    write_tecplot_ascii(p, _ds(GRID_3D))
    ds = read_tecplot_ascii(p)

    assert len(ds.flow) == 0


# ======================================================================
# grid-only binary-direct conversion
# ======================================================================

def test_binary_direct_grid_only(tmp_path: Path) -> None:
    """Binary-direct writer with grid only (no flow dict)."""
    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    write_binary_direct(flow_out, grid_out, _ds(GRID_3D))

    # flow binary should NOT exist
    assert not flow_out.exists()
    # grid binary should exist
    assert grid_out.exists()
    assert grid_out.with_suffix(".cd").exists()


# ======================================================================
# cross-format conversions via dispatcher
# ======================================================================

def test_convert_plot3d_to_hdf5(tmp_path: Path) -> None:
    """Plot3D -> HDF5 via convert()."""
    p3d = tmp_path / "grid.x"
    write_plot3d(p3d, _ds(GRID_3D), binary=True)

    h5 = tmp_path / "grid.h5"
    do_convert(p3d, h5)

    from cfd_io.readers.hdf5 import read_hdf5
    ds = read_hdf5(h5)
    assert ds.grid.x is not None


def test_convert_plot3d_to_split(tmp_path: Path) -> None:
    """Plot3D -> split binary grid-only via convert()."""
    p3d = tmp_path / "grid.x"
    write_plot3d(p3d, _ds(GRID_3D), binary=True)

    flow_out = tmp_path / "grid_flow.s8"
    grid_out = tmp_path / "grid_out.s8"
    do_convert(p3d, flow_out, output_grid=grid_out)

    # grid file should exist
    assert grid_out.exists()


def test_convert_split_to_plot3d(tmp_path: Path) -> None:
    """Split binary grid-only -> Plot3D via convert()."""
    grid_out = tmp_path / "grid.s8"
    flow_out = tmp_path / "flow.s8"
    write_binary_direct(flow_out, grid_out, _ds(GRID_3D))

    p3d = tmp_path / "grid.x"
    do_convert(grid_out, p3d, input_grid=grid_out)

    ds = read_plot3d(p3d)
    assert ds.grid.x is not None


def test_convert_tecplot_to_hdf5(tmp_path: Path) -> None:
    """Tecplot ASCII -> HDF5 via convert()."""
    dat = tmp_path / "data.dat"
    write_tecplot_ascii(dat, _ds(GRID_3D, FLOW))

    h5 = tmp_path / "data.h5"
    do_convert(dat, h5)

    from cfd_io.readers.hdf5 import read_hdf5
    ds = read_hdf5(h5)
    assert ds.grid.x is not None
    assert "uvel" in ds.flow


def test_convert_hdf5_to_tecplot(tmp_path: Path) -> None:
    """HDF5 -> Tecplot ASCII via convert()."""
    from cfd_io.writers.hdf5 import write_hdf5

    h5 = tmp_path / "test.h5"
    write_hdf5(h5, _ds(GRID_3D, FLOW))

    dat = tmp_path / "out.dat"
    do_convert(h5, dat)

    ds = read_tecplot_ascii(dat)
    assert ds.grid.x is not None
    assert "uvel" in ds.flow


def test_convert_tecplot_to_split(tmp_path: Path) -> None:
    """Tecplot ASCII -> split binary via convert()."""
    dat = tmp_path / "data.dat"
    write_tecplot_ascii(dat, _ds(GRID_3D, FLOW))

    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    do_convert(dat, flow_out, output_grid=grid_out)

    ds = read_binary_direct(flow_out, grid_out)
    assert ds.grid.x is not None
    assert "uvel" in ds.flow


# ======================================================================
# Plot3D .q solution file tests
# ======================================================================

# reference freestream conditions for .q test files
_MACH, _ALPHA, _RE, _TIME = 2.5, 5.0, 1e6, 0.0

# 3-D conserved-variable flow arrays
QFLOW_3D = {
    "dens": np.random.rand(NX, NY, NZ),
    "xmom": np.random.rand(NX, NY, NZ),
    "ymom": np.random.rand(NX, NY, NZ),
    "zmom": np.random.rand(NX, NY, NZ),
    "energy": np.random.rand(NX, NY, NZ),
}

# 2-D conserved-variable flow arrays
QFLOW_2D = {
    "dens": np.random.rand(NX, NY, 1),
    "xmom": np.random.rand(NX, NY, 1),
    "ymom": np.random.rand(NX, NY, 1),
    "energy": np.random.rand(NX, NY, 1),
}


# -- helpers to write .q test files ------------------------------------

def _write_q_binary_3d(path: Path) -> None:
    """Write a 3-D binary .q file using FortranBinaryWriter."""
    with FortranBinaryWriter(path) as w:
        w.write_ints([NX, NY, NZ])
        w.write_reals([_MACH, _ALPHA, _RE, _TIME])
        # all flow variables concatenated in one Fortran record
        parts = [QFLOW_3D[n].ravel(order="F") for n in
                 ("dens", "xmom", "ymom", "zmom", "energy")]
        w.write_array_real(np.concatenate(parts), fortran_order=False)


def _write_q_binary_2d(path: Path) -> None:
    """Write a 2-D binary .q file using FortranBinaryWriter."""
    with FortranBinaryWriter(path) as w:
        w.write_ints([NX, NY])
        w.write_reals([_MACH, _ALPHA, _RE, _TIME])
        parts = [QFLOW_2D[n].ravel(order="F") for n in
                 ("dens", "xmom", "ymom", "energy")]
        w.write_array_real(np.concatenate(parts), fortran_order=False)


def _write_q_ascii_3d(path: Path) -> None:
    """Write a 3-D ASCII .q file."""
    with open(path, "w") as fobj:
        fobj.write(f"{NX} {NY} {NZ}\n")
        fobj.write(f"{_MACH} {_ALPHA} {_RE} {_TIME}\n")
        for name in ("dens", "xmom", "ymom", "zmom", "energy"):
            arr = QFLOW_3D[name].ravel(order="F")
            for val in arr:
                fobj.write(f" {val:.15e}")
            fobj.write("\n")


def _write_q_ascii_2d(path: Path) -> None:
    """Write a 2-D ASCII .q file."""
    with open(path, "w") as fobj:
        fobj.write(f"{NX} {NY}\n")
        fobj.write(f"{_MACH} {_ALPHA} {_RE} {_TIME}\n")
        for name in ("dens", "xmom", "ymom", "energy"):
            arr = QFLOW_2D[name].ravel(order="F")
            for val in arr:
                fobj.write(f" {val:.15e}")
            fobj.write("\n")


# -- binary .q tests ---------------------------------------------------

def test_plot3d_flow_binary_3d(tmp_path: Path) -> None:
    """3-D binary .q round-trip."""
    p = tmp_path / "flow.q"
    _write_q_binary_3d(p)
    flow, attrs = read_plot3d_flow_binary(p)

    assert set(flow) == {"dens", "xmom", "ymom", "zmom", "energy"}
    assert np.allclose(flow["dens"], QFLOW_3D["dens"])
    assert np.allclose(flow["energy"], QFLOW_3D["energy"])
    assert attrs["mach"] == _MACH
    assert attrs["alpha"] == _ALPHA
    assert attrs["re"] == _RE


def test_plot3d_flow_binary_2d(tmp_path: Path) -> None:
    """2-D binary .q round-trip."""
    p = tmp_path / "flow.q"
    _write_q_binary_2d(p)
    flow, attrs = read_plot3d_flow_binary(p)

    assert set(flow) == {"dens", "xmom", "ymom", "energy"}
    assert "zmom" not in flow
    assert np.allclose(flow["dens"], QFLOW_2D["dens"])
    assert attrs["mach"] == _MACH


# -- ASCII .q tests ----------------------------------------------------

def test_plot3d_flow_ascii_3d(tmp_path: Path) -> None:
    """3-D ASCII .q round-trip."""
    p = tmp_path / "flow.q"
    _write_q_ascii_3d(p)
    flow, attrs = read_plot3d_flow_ascii(p)

    assert set(flow) == {"dens", "xmom", "ymom", "zmom", "energy"}
    assert np.allclose(flow["dens"], QFLOW_3D["dens"], atol=1e-10)
    assert np.allclose(flow["energy"], QFLOW_3D["energy"], atol=1e-10)
    assert attrs["mach"] == _MACH
    assert attrs["re"] == _RE


def test_plot3d_flow_ascii_2d(tmp_path: Path) -> None:
    """2-D ASCII .q round-trip."""
    p = tmp_path / "flow.q"
    _write_q_ascii_2d(p)
    flow, attrs = read_plot3d_flow_ascii(p)

    assert set(flow) == {"dens", "xmom", "ymom", "energy"}
    assert "zmom" not in flow
    assert np.allclose(flow["dens"], QFLOW_2D["dens"], atol=1e-10)
    assert attrs["alpha"] == _ALPHA


# -- dispatcher tests --------------------------------------------------

def test_plot3d_flow_dispatcher_binary(tmp_path: Path) -> None:
    """Dispatcher auto-detects binary .q file."""
    p = tmp_path / "flow.q"
    _write_q_binary_3d(p)
    ds = read_plot3d_flow(p)

    assert "dens" in ds.flow
    assert ds.attrs["mach"] == _MACH


def test_plot3d_flow_dispatcher_ascii(tmp_path: Path) -> None:
    """Dispatcher auto-detects ASCII .q file."""
    p = tmp_path / "flow.q"
    _write_q_ascii_3d(p)
    ds = read_plot3d_flow(p)

    assert "dens" in ds.flow
    assert ds.attrs["mach"] == _MACH


def test_read_file_q_extension(tmp_path: Path) -> None:
    """read_file dispatches .q files to plot3d_flow reader."""
    p = tmp_path / "flow.q"
    _write_q_ascii_3d(p)
    ds = read_file(p)

    assert "dens" in ds.flow and "energy" in ds.flow
    assert ds.attrs["mach"] == _MACH
