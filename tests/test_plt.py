"""Smoke tests for Tecplot binary (.plt) support via pytecplot.

Requires pytecplot and must be run through tec360-env:
    "/Applications/Tecplot 360 EX 2025 R2/bin/tec360-env" -- python test_plt.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cfd_io.convert_mod import do_convert, read_file, write_file
from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.readers.tecplot_binary import read_tecplot_plt
from cfd_io.writers.hdf5 import write_hdf5
from cfd_io.writers.tecplot_binary import write_tecplot_plt

# Skip entire module when Tecplot 360 native libraries are not available.
# pytecplot may be installed, but the native libs (loaded lazily) are only
# present inside the tec360-env wrapper.
_has_tecplot = False
try:
    import tecplot  # noqa: F401

    tecplot.active_page()  # forces native lib load — fast fail if missing
    _has_tecplot = True
except Exception:
    pass
pytestmark = pytest.mark.skipif(
    not _has_tecplot, reason="Tecplot 360 libraries not available"
)


def make_data():
    ni, nj, nk = 5, 4, 3
    grid = {
        "x": np.random.rand(ni, nj, nk),
        "y": np.random.rand(ni, nj, nk),
        "z": np.random.rand(ni, nj, nk),
    }
    flow = {
        "rho": np.random.rand(ni, nj, nk),
        "uvel": np.random.rand(ni, nj, nk),
    }
    return grid, flow


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


def test_roundtrip():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.plt"
        write_tecplot_plt(p, _ds(grid, flow, {"zone_title": "TestZone"}))
        assert p.exists()

        ds = read_tecplot_plt(p)
        assert ds.grid.shape == (5, 4, 3)
        assert set(ds.flow.keys()) == {"rho", "uvel"}
        np.testing.assert_allclose(ds.grid.x, grid["x"], atol=1e-12)
        np.testing.assert_allclose(ds.flow["rho"].data, flow["rho"], atol=1e-12)
        assert ds.attrs.get("zone_title") == "TestZone"

    print("  PASS: roundtrip")


def test_grid_only():
    grid, _ = make_data()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "grid_only.plt"
        write_tecplot_plt(p, _ds(grid))

        ds = read_tecplot_plt(p)
        assert ds.grid.shape == (5, 4, 3)
        assert len(ds.flow) == 0

    print("  PASS: grid-only")


def test_dispatch():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dispatch.plt"
        write_file(p, _ds(grid, flow, {"zone_title": "Dispatch"}))

        ds = read_file(p)
        np.testing.assert_allclose(ds.grid.y, grid["y"], atol=1e-12)

    print("  PASS: dispatch (read_file/write_file)")


def test_plt_to_hdf5():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        plt_path = Path(td) / "src.plt"
        h5_path = Path(td) / "dst.h5"
        write_tecplot_plt(plt_path, _ds(grid, flow))
        do_convert(plt_path, h5_path)

        ds = read_hdf5(h5_path)
        np.testing.assert_allclose(ds.grid.x, grid["x"], atol=1e-6)

    print("  PASS: cross-format plt -> hdf5")


def test_hdf5_to_plt():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "src.h5"
        plt_path = Path(td) / "dst.plt"
        write_hdf5(h5_path, _ds(grid, flow), dtype="d")
        do_convert(h5_path, plt_path)

        ds = read_tecplot_plt(plt_path)
        np.testing.assert_allclose(ds.grid.z, grid["z"], atol=1e-6)

    print("  PASS: cross-format hdf5 -> plt")


if __name__ == "__main__":
    print("PLT smoke tests:")
    test_roundtrip()
    test_grid_only()
    test_dispatch()
    test_plt_to_hdf5()
    test_hdf5_to_plt()
    print("\nALL PLT TESTS PASSED")
