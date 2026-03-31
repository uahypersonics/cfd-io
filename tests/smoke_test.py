"""Comprehensive smoke test for cfd-io v0.2.0 feature set.

Tests:
    1. Binary-direct round-trip with extended .cd
    2. Float32 round-trip
    3. HDF5 single-timestep round-trip
    4. HDF5 multi-timestep round-trip (reader only)
    5. HDF5 read specific timestep (reader only)
    6. convert split -> HDF5 via dispatcher
    7. convert HDF5 -> split via dispatcher
    8. convert float64 -> float32 via dispatcher
"""

from pathlib import Path

import h5py
import numpy as np

from cfd_io.convert_mod import do_convert, read_file, write_file
from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers.fortran_binary_direct import read_binary_direct, read_header
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.hdf5 import write_hdf5

# -- shared test data --------------------------------------------------
NX, NY, NZ = 10, 8, 3
np.random.seed(42)
GRID = {"x": np.random.rand(NX, NY, NZ), "y": np.random.rand(NX, NY, NZ)}
FLOW = {"uvel": np.random.rand(NX, NY, NZ), "temp": np.random.rand(NX, NY, NZ)}
GRID_2D = {"x": np.random.rand(NX, NY), "y": np.random.rand(NX, NY)}
FLOW_2D = {"uvel": np.random.rand(NX, NY), "temp": np.random.rand(NX, NY)}


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


def test_binary_direct_roundtrip_extended_cd(tmp_path: Path) -> None:
    """Binary-direct round-trip with extended .cd (var names, info lines, timesteps)."""
    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    write_binary_direct(
        flow_out, grid_out,
        _ds(GRID, FLOW, {"info_lines": ["Test run", "M=6.0"], "timesteps": [100, 200]}),
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
    ds = read_binary_direct(flow_out, grid_out)
    assert np.allclose(GRID["x"], ds.grid.x)
    assert np.allclose(FLOW["uvel"], ds.flow["uvel"].data)


def test_float32_roundtrip(tmp_path: Path) -> None:
    """Float32 round-trip (lossy)."""
    flow_out = tmp_path / "flow.s4"
    grid_out = tmp_path / "grid.s4"
    write_binary_direct(flow_out, grid_out, _ds(GRID, FLOW))

    header = read_header(flow_out.with_suffix(".cd"))
    assert header.precision == "float32"
    assert header.dtype == np.float32

    ds = read_binary_direct(flow_out, grid_out)
    assert np.allclose(GRID["x"], ds.grid.x, atol=1e-6)
    assert np.allclose(FLOW["uvel"], ds.flow["uvel"].data, atol=1e-6)


def test_hdf5_single_timestep(tmp_path: Path) -> None:
    """HDF5 single-timestep round-trip."""
    h5 = tmp_path / "test.h5"
    write_hdf5(h5, _ds(GRID, FLOW, {"mach": 6.0}))

    ds = read_hdf5(h5)
    assert ds.grid.x is not None
    assert "uvel" in ds.flow
    assert ds.attrs.get("mach") == 6.0
    assert isinstance(ds.flow["uvel"].data, np.ndarray)


def test_hdf5_multi_timestep(tmp_path: Path) -> None:
    """HDF5 multi-timestep round-trip with timestep groups."""
    h5 = tmp_path / "multi.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("grid")
        g.create_dataset("x", data=GRID["x"])
        g.create_dataset("y", data=GRID["y"])
        fg = f.create_group("flow")
        ts1 = fg.create_group("00001")
        ts1.create_dataset("uvel", data=np.ones((NX, NY, NZ)))
        ts1.create_dataset("temp", data=np.full((NX, NY, NZ), 300.0))
        ts5 = fg.create_group("00005")
        ts5.create_dataset("uvel", data=np.ones((NX, NY, NZ)) * 2)
        ts5.create_dataset("temp", data=np.full((NX, NY, NZ), 600.0))
        f.attrs["mach"] = 4.0

    ds = read_hdf5(h5)
    # Dataset is single snapshot -- reader picks first timestep
    assert "uvel" in ds.flow
    assert np.allclose(ds.flow["uvel"].data, 1.0)
    assert ds.attrs.get("timesteps") == [1, 5]


def test_hdf5_read_specific_timestep(tmp_path: Path) -> None:
    """HDF5: read a specific timestep returns single-snapshot Dataset."""
    h5 = tmp_path / "multi.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("grid")
        g.create_dataset("x", data=GRID["x"])
        g.create_dataset("y", data=GRID["y"])
        fg = f.create_group("flow")
        ts1 = fg.create_group("00001")
        ts1.create_dataset("uvel", data=np.ones((NX, NY, NZ)))
        ts5 = fg.create_group("00005")
        ts5.create_dataset("uvel", data=np.ones((NX, NY, NZ)) * 2)

    ds = read_hdf5(h5, timestep=5)
    assert isinstance(ds.flow["uvel"].data, np.ndarray)
    assert np.allclose(ds.flow["uvel"].data, 2.0)


def test_convert_split_to_hdf5(tmp_path: Path) -> None:
    """Convert split binary -> HDF5 via dispatcher."""
    flow_out = tmp_path / "flow.s8"
    grid_out = tmp_path / "grid.s8"
    write_binary_direct(flow_out, grid_out, _ds(GRID, FLOW))

    h5_out = tmp_path / "converted.h5"
    do_convert(flow_out, h5_out, input_grid=grid_out)

    ds = read_hdf5(h5_out)
    assert ds.grid.x is not None
    assert "uvel" in ds.flow


def test_convert_hdf5_to_split(tmp_path: Path) -> None:
    """Convert HDF5 -> split binary via dispatcher."""
    h5 = tmp_path / "test.h5"
    write_hdf5(h5, _ds(GRID, FLOW))

    flow_out = tmp_path / "back.s8"
    grid_out = tmp_path / "back_grid.s8"
    do_convert(h5, flow_out, output_grid=grid_out)

    ds = read_binary_direct(flow_out, grid_out)
    assert ds.grid.x is not None


def test_convert_f64_to_f32(tmp_path: Path) -> None:
    """Convert float64 -> float32 via read_file/write_file dispatcher."""
    flow_f64 = tmp_path / "flow.s8"
    grid_f64 = tmp_path / "grid.s8"
    write_binary_direct(flow_f64, grid_f64, _ds(GRID, FLOW))

    flow_f32 = tmp_path / "down.s4"
    grid_f32 = tmp_path / "down_grid.s4"
    ds = read_file(flow_f64, grid_file=grid_f64)
    write_file(flow_f32, ds, grid_file=grid_f32)

    header = read_header(flow_f32.with_suffix(".cd"))
    assert header.dtype == np.float32


def test_write_split_with_2d_arrays_promotes_to_nz1(tmp_path: Path) -> None:
    """Split writer accepts 2-D arrays by promoting them to nz=1."""
    flow_out = tmp_path / "flow_2d.s8"
    grid_out = tmp_path / "grid_2d.s8"

    # write 2-D dataset to split format
    write_binary_direct(flow_out, grid_out, _ds(GRID_2D, FLOW_2D))

    # check header dimensions reflect nz=1 promotion
    flow_header = read_header(flow_out.with_suffix(".cd"))
    assert flow_header.nx == NX
    assert flow_header.ny == NY
    assert flow_header.nz == 1

    # check round-trip values with explicit singleton z-axis
    ds = read_binary_direct(flow_out, grid_out)
    assert ds.grid.x.shape == (NX, NY, 1)
    assert np.allclose(ds.grid.x[:, :, 0], GRID_2D["x"])
    assert np.allclose(ds.flow["uvel"].data[:, :, 0], FLOW_2D["uvel"])


def test_convert_hdf5_2d_to_split(tmp_path: Path) -> None:
    """Dispatcher converts 2-D HDF5 to split format by writing nz=1."""
    h5 = tmp_path / "test_2d.h5"
    write_hdf5(h5, _ds(GRID_2D, FLOW_2D))

    flow_out = tmp_path / "back_2d.s8"
    grid_out = tmp_path / "back_grid_2d.s8"
    do_convert(h5, flow_out, output_grid=grid_out)

    # check converted split data is readable and promoted to 3-D
    ds = read_binary_direct(flow_out, grid_out)
    assert ds.grid.x.shape == (NX, NY, 1)
    assert np.allclose(ds.grid.x[:, :, 0], GRID_2D["x"])
    assert np.allclose(ds.flow["temp"].data[:, :, 0], FLOW_2D["temp"])
