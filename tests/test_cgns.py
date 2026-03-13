"""Tests for the CGNS reader."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from cfd_io.dataset import UnstructuredGrid
from cfd_io.readers.cgns import (
    CGNS_MIXED,
    read_cgns,
)

# ── helpers to build minimal CGNS files ──────────────────────────────

def _set_label(group: h5py.Group, label: str) -> None:
    group.attrs["label"] = label.encode()


def _write_string(group: h5py.Group, text: str) -> None:
    group.create_dataset(" data", data=np.frombuffer(text.encode("ascii"), dtype=np.int8))


def _write_data(group: h5py.Group, arr: np.ndarray) -> None:
    group.create_dataset(" data", data=arr)


def _make_cgns_file(
    path: Path,
    *,
    n_pts: int = 8,
    points: np.ndarray | None = None,
    etype: int = 17,  # HEXA_8
    conn_1based: np.ndarray | None = None,
    mixed: bool = False,
    flow_vars: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a minimal CGNS/HDF5 file for testing."""
    if points is None:
        points = np.random.rand(n_pts, 3)
    else:
        n_pts = len(points)

    with h5py.File(path, "w") as f:
        # root markers
        f.attrs["label"] = b"Root"
        f.attrs["name"] = b"HDF5 MotherNode"
        f.attrs["type"] = b"MT"

        # CGNSLibraryVersion
        ver = f.create_group("CGNSLibraryVersion")
        _set_label(ver, "CGNSLibraryVersion_t")
        _write_data(ver, np.array([4.3], dtype=np.float32))

        # Base
        base = f.create_group("Base")
        _set_label(base, "CGNSBase_t")
        _write_data(base, np.array([2, 3], dtype=np.int32))

        # Zone
        zone = base.create_group("Zone")
        _set_label(zone, "Zone_t")

        if mixed and conn_1based is not None:
            pass  # n_cells calculated below from offsets
        elif conn_1based is not None and not mixed:
            pass  # uniform element type
        else:
            conn_1based = np.arange(1, n_pts + 1, dtype=np.int64)

        # ZoneType
        zt = zone.create_group("ZoneType")
        _set_label(zt, "ZoneType_t")
        _write_string(zt, "Unstructured")

        # GridCoordinates
        gc = zone.create_group("GridCoordinates")
        _set_label(gc, "GridCoordinates_t")
        for i, name in enumerate(["CoordinateX", "CoordinateY", "CoordinateZ"]):
            da = gc.create_group(name)
            _set_label(da, "DataArray_t")
            _write_data(da, points[:, i])

        # Elements
        sec = zone.create_group("Elements")
        _set_label(sec, "Elements_t")

        if mixed:
            _write_data(sec, np.array([CGNS_MIXED, 0], dtype=np.int32))
            _write_data_elem_conn(sec, conn_1based)
        else:
            _write_data(sec, np.array([etype, 0], dtype=np.int32))
            ec = sec.create_group("ElementConnectivity")
            _set_label(ec, "DataArray_t")
            _write_data(ec, conn_1based)

        # compute n_cells for zone data
        if mixed:
            # n_cells from offsets already written
            offsets = sec["ElementStartOffset"][" data"][:]
            n_cells_val = len(offsets) - 1
        else:
            from cfd_io.readers.cgns import CGNS_ELEMENT_NODES

            npn = CGNS_ELEMENT_NODES[etype]
            n_cells_val = len(conn_1based) // npn

        zone_data = np.array([[n_pts], [n_cells_val], [0]], dtype=np.int64)
        _write_data(zone, zone_data)

        # ElementRange
        er = sec.create_group("ElementRange")
        _set_label(er, "IndexRange_t")
        _write_data(er, np.array([1, n_cells_val], dtype=np.int64))

        # FlowSolution
        if flow_vars:
            fs = zone.create_group("FlowSolution")
            _set_label(fs, "FlowSolution_t")
            for vname, varr in flow_vars.items():
                da = fs.create_group(vname)
                _set_label(da, "DataArray_t")
                _write_data(da, varr)

    return path


def _write_data_elem_conn(sec: h5py.Group, raw_conn: np.ndarray) -> None:
    """Write mixed-element connectivity with offsets."""
    ec = sec.create_group("ElementConnectivity")
    _set_label(ec, "DataArray_t")
    _write_data(ec, raw_conn)

    # Build offsets from type codes in raw_conn
    from cfd_io.readers.cgns import CGNS_ELEMENT_NODES

    offsets = [0]
    i = 0
    while i < len(raw_conn):
        etype = int(raw_conn[i])
        npn = CGNS_ELEMENT_NODES[etype]
        i += 1 + npn
        offsets.append(i)

    eso = sec.create_group("ElementStartOffset")
    _set_label(eso, "DataArray_t")
    _write_data(eso, np.array(offsets, dtype=np.int64))


def _set_label(group, label):
    group.attrs["label"] = label.encode()


# ── unit tests ────────────────────────────────────────────────────────

class TestReadCGNSUniform:
    """Uniform element type (QUAD_4)."""

    def test_reads_quad_mesh(self, tmp_path):
        # 4 points, 1 quad
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        conn = np.array([1, 2, 3, 4], dtype=np.int64)  # 1-based

        path = _make_cgns_file(
            tmp_path / "quad.cgns",
            points=pts,
            etype=7,  # QUAD_4
            conn_1based=conn,
            flow_vars={"Pressure": np.array([100.0, 200.0, 300.0, 400.0])},
        )

        ds = read_cgns(path)
        assert isinstance(ds.grid, UnstructuredGrid)
        assert ds.grid.points.shape == (4, 3)
        np.testing.assert_array_equal(ds.grid.cell_types, [7])
        np.testing.assert_array_equal(ds.grid.offsets, [0, 4])  # 1 quad, 4 nodes
        assert "pres" in ds.flow  # normalized from "Pressure"
        np.testing.assert_allclose(ds.flow["pres"].data, [100, 200, 300, 400])
        assert ds.attrs["format"] == "cgns"
        assert ds.attrs["n_vertices"] == 4
        assert ds.attrs["n_cells"] == 1

    def test_reads_tri_mesh(self, tmp_path):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
        conn = np.array([1, 2, 3], dtype=np.int64)

        path = _make_cgns_file(
            tmp_path / "tri.cgns",
            points=pts,
            etype=5,  # TRI_3
            conn_1based=conn,
        )

        ds = read_cgns(path)
        assert ds.grid.points.shape == (3, 3)
        np.testing.assert_array_equal(ds.grid.cell_types, [5])
        np.testing.assert_array_equal(ds.grid.offsets, [0, 3])  # 1 tri, 3 nodes
        assert ds.attrs["n_cells"] == 1


class TestReadCGNSMixed:
    """MIXED element type with embedded type codes."""

    def test_reads_mixed_tri_quad(self, tmp_path):
        # 5 points: a quad (0,1,2,3) and a tri (2,3,4)
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 2, 0]],
            dtype=np.float64,
        )
        # MIXED connectivity: [type, nodes...] for each element
        raw_conn = np.array([7, 1, 2, 3, 4, 5, 3, 4, 5], dtype=np.int64)

        path = _make_cgns_file(
            tmp_path / "mixed.cgns",
            points=pts,
            mixed=True,
            conn_1based=raw_conn,
        )

        ds = read_cgns(path)
        assert ds.grid.points.shape == (5, 3)
        np.testing.assert_array_equal(ds.grid.cell_types, [7, 5])
        # connectivity should be 0-based, type codes stripped
        expected_conn = np.array([0, 1, 2, 3, 2, 3, 4])
        np.testing.assert_array_equal(ds.grid.connectivity, expected_conn)
        np.testing.assert_array_equal(ds.grid.offsets, [0, 4, 7])  # quad(4) + tri(3)
        assert ds.attrs["n_cells"] == 2


class TestReadCGNSFlowNormalization:
    """Flow variable names are normalized via _aliases."""

    def test_cgns_variable_names_normalized(self, tmp_path):
        pts = np.random.rand(4, 3)
        conn = np.array([1, 2, 3, 4], dtype=np.int64)

        path = _make_cgns_file(
            tmp_path / "vars.cgns",
            points=pts,
            etype=7,
            conn_1based=conn,
            flow_vars={
                "Pressure": np.ones(4),
                "Temperature": np.ones(4) * 300,
                "Density": np.ones(4) * 1.2,
                "Mach": np.ones(4) * 2.0,
                "CoefPressure": np.ones(4) * 0.5,
                "TemperatureStagnation": np.ones(4) * 500,
            },
        )

        ds = read_cgns(path)
        assert "pres" in ds.flow
        assert "temp" in ds.flow
        assert "dens" in ds.flow
        assert "mach" in ds.flow
        assert "cp" in ds.flow
        assert "temp_stag" in ds.flow


class TestReadCGNSErrors:
    """Error handling."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_cgns("/nonexistent/file.cgns")

    def test_no_base(self, tmp_path):
        path = tmp_path / "empty.cgns"
        with h5py.File(path, "w"):
            pass
        with pytest.raises(ValueError, match="no CGNSBase_t"):
            read_cgns(path)


# ── integration test with real file ───────────────────────────────────

REAL_FILE = Path("/Users/chader/cfd-io/sandbox/cgns_test/5mm_case.cgns")


@pytest.mark.skipif(not REAL_FILE.is_file(), reason="real CGNS file not available")
class TestReadCGNSReal:
    """Smoke tests against the real 5mm_case.cgns file."""

    def test_reads_real_file(self):
        ds = read_cgns(REAL_FILE)

        assert isinstance(ds.grid, UnstructuredGrid)
        assert ds.grid.points.shape == (2395796, 3)
        assert len(ds.grid.cell_types) == 2282100
        assert len(ds.grid.offsets) == 2282101  # n_cells + 1
        assert ds.grid.offsets[0] == 0
        assert ds.grid.offsets[-1] == len(ds.grid.connectivity)
        assert ds.attrs["format"] == "cgns"

    def test_flow_variables_present(self):
        ds = read_cgns(REAL_FILE)

        # these should be normalized from the CGNS names
        assert "pres" in ds.flow
        assert "dens" in ds.flow
        assert "mach" in ds.flow
        assert "temp" in ds.flow
        assert "uvel" in ds.flow  # Axial_Velocity
        assert "cp" in ds.flow  # CoefPressure

    def test_read_file_dispatch(self):
        """read_file() should dispatch .cgns to read_cgns."""
        from cfd_io.convert_mod import read_file

        ds = read_file(REAL_FILE)
        assert isinstance(ds.grid, UnstructuredGrid)
        assert ds.attrs["format"] == "cgns"
