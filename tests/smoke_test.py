"""Comprehensive smoke test for cfd-io v0.1.0 feature set.

Tests:
    1. Binary-direct round-trip with extended .cd
    2. Float32 round-trip
    3. HDF5 single-timestep round-trip
    4. HDF5 multi-timestep round-trip
    5. HDF5 read specific timestep
    6. convert split -> HDF5 via dispatcher
    7. convert HDF5 -> split via dispatcher
    8. convert float64 -> float32 via dispatcher
"""

from pathlib import Path

import numpy as np

from cfd_io.convert_mod import do_convert, read_file, write_file
from cfd_io.readers.fortran_binary_direct import read_binary_direct, read_header
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.hdf5 import write_hdf5

# -- shared test data --------------------------------------------------
NX, NY, NZ = 10, 8, 3
np.random.seed(42)
GRID = {"x": np.random.rand(NX, NY, NZ), "y": np.random.rand(NX, NY, NZ)}
FLOW = {"uvel": np.random.rand(NX, NY, NZ), "temp": np.random.rand(NX, NY, NZ)}


def test_binary_direct_roundtrip_extended_cd(tmp_path: Path) -> None:
    """Binary-direct round-trip with extended .cd (var names, info lines, timesteps)."""
    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    write_binary_direct(
        flow_out, grid_out, GRID, FLOW,
        attrs={"info_lines": ["Test run", "M=6.0"], "timesteps": [100, 200]},
    )

    # verify .cd content
    cd_text = flow_out.with_suffix(".cd").read_text()
    assert "uvel" in cd_text
    assert "Test run" in cd_text
    assert "100" in cd_text

    # verify parsed header
    header = read_header(flow_out.with_suffix(".cd"))
    assert header.nx == NX
    assert header.ny == NY
    assert header.nz == NZ
    assert header.var_names == ["uvel", "temp"]
    assert header.info_lines == ["Test run", "M=6.0"]
    assert header.timesteps == [100, 200]
    assert header.precision == "float64"

    # verify data round-trip
    g, f, _a = read_binary_direct(flow_out, grid_out)
    assert np.allclose(GRID["x"], g["x"])
    assert np.allclose(FLOW["uvel"], f["uvel"])


def test_float32_roundtrip(tmp_path: Path) -> None:
    """Float32 round-trip (lossy)."""
    flow_out = tmp_path / "flow.s4"
    grid_out = tmp_path / "grid.s4"
    write_binary_direct(flow_out, grid_out, GRID, FLOW)

    header = read_header(flow_out.with_suffix(".cd"))
    assert header.precision == "float32"
    assert header.dtype == np.float32

    g, f, _ = read_binary_direct(flow_out, grid_out)
    assert np.allclose(GRID["x"], g["x"], atol=1e-6)
    assert np.allclose(FLOW["uvel"], f["uvel"], atol=1e-6)


def test_hdf5_single_timestep(tmp_path: Path) -> None:
    """HDF5 single-timestep round-trip."""
    h5 = tmp_path / "test.h5"
    write_hdf5(h5, GRID, FLOW, attrs={"mach": 6.0})

    _g, f, a = read_hdf5(h5)
    assert "x" in _g
    assert "uvel" in f
    assert a.get("mach") == 6.0
    # single timestep -> flat dict (not nested)
    assert isinstance(f["uvel"], np.ndarray)


def test_hdf5_multi_timestep(tmp_path: Path) -> None:
    """HDF5 multi-timestep round-trip with timestep groups."""
    multi_flow = {
        1: {"uvel": np.ones((NX, NY, NZ)), "temp": np.full((NX, NY, NZ), 300.0)},
        5: {"uvel": np.ones((NX, NY, NZ)) * 2, "temp": np.full((NX, NY, NZ), 600.0)},
    }
    h5 = tmp_path / "multi.h5"
    write_hdf5(h5, GRID, multi_flow, attrs={"mach": 4.0})

    _g, f, a = read_hdf5(h5)
    # multi-timestep -> {int: dict}
    assert 1 in f and 5 in f
    assert np.allclose(f[1]["uvel"], 1.0)
    assert np.allclose(f[5]["temp"], 600.0)
    assert a.get("timesteps") == [1, 5]


def test_hdf5_read_specific_timestep(tmp_path: Path) -> None:
    """HDF5: read a specific timestep returns flat dict."""
    multi_flow = {
        1: {"uvel": np.ones((NX, NY, NZ))},
        5: {"uvel": np.ones((NX, NY, NZ)) * 2},
    }
    h5 = tmp_path / "multi.h5"
    write_hdf5(h5, GRID, multi_flow)

    _, f5, _ = read_hdf5(h5, timestep=5)
    assert isinstance(f5["uvel"], np.ndarray)
    assert np.allclose(f5["uvel"], 2.0)


def test_convert_split_to_hdf5(tmp_path: Path) -> None:
    """Convert split binary -> HDF5 via dispatcher."""
    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    write_binary_direct(flow_out, grid_out, GRID, FLOW)

    h5_out = tmp_path / "converted.h5"
    do_convert(flow_out, h5_out, input_grid=grid_out)

    g, f, _ = read_hdf5(h5_out)
    assert "x" in g
    assert "uvel" in f


def test_convert_hdf5_to_split(tmp_path: Path) -> None:
    """Convert HDF5 -> split binary via dispatcher."""
    h5 = tmp_path / "test.h5"
    write_hdf5(h5, GRID, FLOW)

    flow_out = tmp_path / "back.s8"
    grid_out = tmp_path / "back_grid.s8"
    do_convert(h5, flow_out, output_grid=grid_out)

    g, _f, _ = read_binary_direct(flow_out, grid_out)
    assert "x" in g


def test_convert_f64_to_f32(tmp_path: Path) -> None:
    """Convert float64 -> float32 via read_file/write_file dispatcher."""
    flow_f64 = tmp_path / "flow.s8"
    grid_f64 = tmp_path / "grid.s8"
    write_binary_direct(flow_f64, grid_f64, GRID, FLOW)

    flow_f32 = tmp_path / "down.s4"
    grid_f32 = tmp_path / "down_grid.s4"
    g, f, a = read_file(flow_f64, grid_file=grid_f64)
    write_file(flow_f32, g, f, a, grid_file=grid_f32)

    header = read_header(flow_f32.with_suffix(".cd"))
    assert header.dtype == np.float32
