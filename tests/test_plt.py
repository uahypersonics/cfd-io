"""Smoke tests for Tecplot binary (.plt) support via pytecplot.

Requires pytecplot and must be run through tec360-env:
    "/Applications/Tecplot 360 EX 2025 R2/bin/tec360-env" -- python test_plt.py
"""

import tempfile
from pathlib import Path

import numpy as np

from cfd_io.convert_mod import do_convert, read_file, write_file
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.readers.tecplot_binary import read_tecplot_plt
from cfd_io.writers.hdf5 import write_hdf5
from cfd_io.writers.tecplot_binary import write_tecplot_plt


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


def test_roundtrip():
    grid, flow = make_data()
    attrs = {"zone_title": "TestZone"}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.plt"
        write_tecplot_plt(p, grid, flow, attrs)
        assert p.exists()

        g2, f2, a2 = read_tecplot_plt(p)
        assert set(g2.keys()) == {"x", "y", "z"}
        assert set(f2.keys()) == {"rho", "uvel"}
        assert g2["x"].shape == (5, 4, 3)
        np.testing.assert_allclose(g2["x"], grid["x"], atol=1e-12)
        np.testing.assert_allclose(f2["rho"], flow["rho"], atol=1e-12)
        assert a2.get("zone_title") == "TestZone"

    print("  PASS: roundtrip")


def test_grid_only():
    grid, _ = make_data()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "grid_only.plt"
        write_tecplot_plt(p, grid, flow=None, attrs=None)

        g3, f3, _ = read_tecplot_plt(p)
        assert set(g3.keys()) == {"x", "y", "z"}
        assert len(f3) == 0

    print("  PASS: grid-only")


def test_dispatch():
    grid, flow = make_data()
    attrs = {"zone_title": "Dispatch"}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dispatch.plt"
        write_file(p, grid, flow, attrs)

        g4, _f4, _ = read_file(p)
        np.testing.assert_allclose(g4["y"], grid["y"], atol=1e-12)

    print("  PASS: dispatch (read_file/write_file)")


def test_plt_to_hdf5():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        plt_path = Path(td) / "src.plt"
        h5_path = Path(td) / "dst.h5"
        write_tecplot_plt(plt_path, grid, flow)
        do_convert(plt_path, h5_path)

        g5, _f5, _ = read_hdf5(h5_path)
        np.testing.assert_allclose(g5["x"], grid["x"], atol=1e-6)

    print("  PASS: cross-format plt -> hdf5")


def test_hdf5_to_plt():
    grid, flow = make_data()

    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "src.h5"
        plt_path = Path(td) / "dst.plt"
        write_hdf5(h5_path, grid, flow, dtype="d")
        do_convert(h5_path, plt_path)

        g6, _f6, _ = read_tecplot_plt(plt_path)
        np.testing.assert_allclose(g6["z"], grid["z"], atol=1e-6)

    print("  PASS: cross-format hdf5 -> plt")


if __name__ == "__main__":
    print("PLT smoke tests:")
    test_roundtrip()
    test_grid_only()
    test_dispatch()
    test_plt_to_hdf5()
    test_hdf5_to_plt()
    print("\nALL PLT TESTS PASSED")
