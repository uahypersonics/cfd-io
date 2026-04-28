"""Tests for the SU2 mesh writer.

Covers:
- Gold-reference conversion test (3×2 grid, hand-verified)
- Orientation tests (adjacent cells share nodes, boundary attachment)
- Negative tests (missing markers, wrong schema, grid too small, …)
- Serializer tests (file format, section counts, coordinate format)
- Duplicate marker merge test
- 3-D conversion smoke test
"""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.writers.su2 import (
    structured_to_unstructured,
    write_su2,
)

# =====================================================================
#  Fixtures
# =====================================================================

def _make_grid_3x2() -> StructuredGrid:
    """3×2×1 structured grid with integer coordinates.

    Node layout (flat index = i + ni * j):

        j=1:  3 --- 4 --- 5
              | Q0  | Q1  |
        j=0:  0 --- 1 --- 2
              i=0   i=1   i=2
    """
    x = np.zeros((3, 2, 1))
    x[0, :, :] = 0.0
    x[1, :, :] = 1.0
    x[2, :, :] = 2.0

    y = np.zeros((3, 2, 1))
    y[:, 0, :] = 0.0
    y[:, 1, :] = 1.0

    z = np.zeros((3, 2, 1))
    return StructuredGrid(x=x, y=y, z=z)


def _make_markers_3x2() -> dict[str, str]:
    return {
        "imin": "inlet",
        "imax": "outlet",
        "jmin": "wall",
        "jmax": "farfield",
    }


def _make_dataset_3x2() -> Dataset:
    return Dataset(
        grid=_make_grid_3x2(),
        flow={},
        attrs={
            "markers": {
                "structured": _make_markers_3x2(),
            }
        },
    )


def _make_grid_3d(ni=3, nj=3, nk=3) -> StructuredGrid:
    """Simple 3-D structured grid with integer coordinates."""
    xs = np.arange(ni, dtype=float)
    ys = np.arange(nj, dtype=float)
    zs = np.arange(nk, dtype=float)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return StructuredGrid(x=x, y=y, z=z)


def _make_markers_3d() -> dict[str, str]:
    return {
        "imin": "inlet",
        "imax": "outlet",
        "jmin": "wall",
        "jmax": "farfield",
        "kmin": "symmetry",
        "kmax": "top",
    }


# =====================================================================
#  Gold-reference tests (3×2 grid)
# =====================================================================

class TestGoldReference2D:
    """Verify exact conversion against hand-computed reference."""

    def setup_method(self):
        grid = _make_grid_3x2()
        markers = _make_markers_3x2()
        self.points, self.elements, self.boundaries = (
            structured_to_unstructured(grid, markers)
        )

    def test_point_count(self):
        assert self.points.shape == (6, 2)

    def test_point_coordinates(self):
        expected = np.array([
            [0.0, 0.0],  # node 0  (i=0, j=0)
            [1.0, 0.0],  # node 1  (i=1, j=0)
            [2.0, 0.0],  # node 2  (i=2, j=0)
            [0.0, 1.0],  # node 3  (i=0, j=1)
            [1.0, 1.0],  # node 4  (i=1, j=1)
            [2.0, 1.0],  # node 5  (i=2, j=1)
        ])
        np.testing.assert_array_equal(self.points, expected)

    def test_element_count(self):
        assert self.elements.shape == (2, 4)

    def test_element_connectivity(self):
        # Quad 0: (i=0,j=0)→(i=1,j=0)→(i=1,j=1)→(i=0,j=1) = 0,1,4,3
        # Quad 1: (i=1,j=0)→(i=2,j=0)→(i=2,j=1)→(i=1,j=1) = 1,2,5,4
        expected = np.array([
            [0, 1, 4, 3],
            [1, 2, 5, 4],
        ])
        np.testing.assert_array_equal(self.elements, expected)

    def test_boundary_marker_keys(self):
        # Canonical side order: imin→inlet, imax→outlet, jmin→wall, jmax→farfield
        assert list(self.boundaries.keys()) == [
            "inlet", "outlet", "wall", "farfield"
        ]

    def test_boundary_imin(self):
        # imin (i=0): segment (0, 3)
        np.testing.assert_array_equal(
            self.boundaries["inlet"], np.array([[0, 3]])
        )

    def test_boundary_imax(self):
        # imax (i=2): segment (2, 5)
        np.testing.assert_array_equal(
            self.boundaries["outlet"], np.array([[2, 5]])
        )

    def test_boundary_jmin(self):
        # jmin (j=0): segments (0,1), (1,2)
        np.testing.assert_array_equal(
            self.boundaries["wall"], np.array([[0, 1], [1, 2]])
        )

    def test_boundary_jmax(self):
        # jmax (j=1): segments (3,4), (4,5)
        np.testing.assert_array_equal(
            self.boundaries["farfield"], np.array([[3, 4], [4, 5]])
        )


# =====================================================================
#  Orientation tests
# =====================================================================

class TestOrientation2D:
    """Verify cell adjacency and boundary attachment."""

    def setup_method(self):
        grid = _make_grid_3x2()
        markers = _make_markers_3x2()
        self.points, self.elements, self.boundaries = (
            structured_to_unstructured(grid, markers)
        )

    def test_adjacent_quads_share_two_nodes(self):
        """Adjacent quads must share exactly 2 nodes (one edge)."""
        q0 = set(self.elements[0])
        q1 = set(self.elements[1])
        shared = q0 & q1
        assert len(shared) == 2

    def test_boundary_nodes_on_correct_side(self):
        """Boundary segments for jmin must use nodes with y=0."""
        wall_nodes = set(self.boundaries["wall"].ravel())
        for nid in wall_nodes:
            assert self.points[nid, 1] == 0.0, (
                f"node {nid} on 'wall' boundary has y={self.points[nid, 1]}"
            )

    def test_boundary_nodes_farfield_on_correct_side(self):
        """Boundary segments for jmax must use nodes with y=1."""
        ff_nodes = set(self.boundaries["farfield"].ravel())
        for nid in ff_nodes:
            assert self.points[nid, 1] == 1.0

    def test_boundary_nodes_inlet_on_correct_side(self):
        """Boundary segments for imin must use nodes with x=0."""
        inlet_nodes = set(self.boundaries["inlet"].ravel())
        for nid in inlet_nodes:
            assert self.points[nid, 0] == 0.0

    def test_boundary_nodes_outlet_on_correct_side(self):
        """Boundary segments for imax must use nodes with x=2."""
        outlet_nodes = set(self.boundaries["outlet"].ravel())
        for nid in outlet_nodes:
            assert self.points[nid, 0] == 2.0

    def test_all_boundary_segments_are_edges_of_volume_elements(self):
        """Every boundary segment must be an edge of some volume element."""
        # Collect all edges from volume elements
        element_edges = set()
        for elem in self.elements:
            for k in range(4):
                edge = frozenset([int(elem[k]), int(elem[(k + 1) % 4])])
                element_edges.add(edge)

        # Check every boundary segment
        for tag, belems in self.boundaries.items():
            for seg in belems:
                edge = frozenset([int(seg[0]), int(seg[1])])
                assert edge in element_edges, (
                    f"boundary segment {seg} (marker={tag}) is not "
                    f"an edge of any volume element"
                )


# =====================================================================
#  Negative tests
# =====================================================================

class TestNegative:
    """Error handling for invalid inputs."""

    def test_missing_marker_key(self):
        grid = _make_grid_3x2()
        incomplete = {"imin": "inlet", "imax": "outlet", "jmin": "wall"}
        with pytest.raises(ValueError, match="missing required marker keys"):
            structured_to_unstructured(grid, incomplete)

    def test_flat_markers_schema(self):
        """attrs['markers'] must have a 'structured' sub-key."""
        ds = Dataset(
            grid=_make_grid_3x2(),
            attrs={"markers": {"imin": "inlet"}},
        )
        with pytest.raises(TypeError, match="structured"):
            write_su2("/tmp/bad.su2", ds)

    def test_markers_not_a_dict(self):
        ds = Dataset(
            grid=_make_grid_3x2(),
            attrs={"markers": "bad"},
        )
        with pytest.raises(TypeError, match="must be a dict"):
            write_su2("/tmp/bad.su2", ds)

    def test_no_markers_key(self):
        """Missing 'markers' key raises with a helpful message."""
        ds = Dataset(grid=_make_grid_3x2(), attrs={})
        with pytest.raises(TypeError, match="missing 'markers'"):
            write_su2("/tmp/bad.su2", ds)

    def test_grid_too_small(self):
        """A 1×2 grid cannot form cells in the i-direction."""
        x = np.zeros((1, 2, 1))
        y = np.zeros((1, 2, 1))
        y[:, 1, :] = 1.0
        z = np.zeros((1, 2, 1))
        grid = StructuredGrid(x=x, y=y, z=z)
        # ndim=1 (only j has size > 1)
        with pytest.raises(ValueError, match="at least a 2-D grid"):
            structured_to_unstructured(grid, {"jmin": "a", "jmax": "b"})

    def test_1d_grid_rejected(self):
        """A purely 1-D grid (only one axis > 1) is not supported."""
        x = np.arange(5, dtype=float).reshape(5, 1, 1)
        y = np.zeros((5, 1, 1))
        z = np.zeros((5, 1, 1))
        grid = StructuredGrid(x=x, y=y, z=z)
        with pytest.raises(ValueError, match="at least a 2-D"):
            structured_to_unstructured(grid, {"imin": "a", "imax": "b"})

    def test_2d_grid_with_3d_marker_keys(self):
        """2-D grid should reject kmin/kmax marker keys (not required)."""
        grid = _make_grid_3x2()
        _markers_with_k = {
            "imin": "inlet", "imax": "outlet",
            "jmin": "wall", "jmax": "farfield",
            "kmin": "sym", "kmax": "top",
        }
        # This should succeed since extra keys are ignored
        # but if only kmin/kmax are provided without i/j, it should fail
        markers_only_k = {"kmin": "sym", "kmax": "top"}
        with pytest.raises(ValueError, match="missing required marker keys"):
            structured_to_unstructured(grid, markers_only_k)

    def test_not_a_structured_grid(self):
        with pytest.raises(TypeError, match="StructuredGrid"):
            structured_to_unstructured("not a grid", {"imin": "a"})

    def test_unstructured_grid_not_implemented(self):
        from cfd_io.dataset import UnstructuredGrid
        ug = UnstructuredGrid(
            points=np.zeros((3, 2)),
            connectivity=np.array([0, 1, 2]),
            offsets=np.array([0, 3]),
            cell_types=np.array([5]),
        )
        ds = Dataset(grid=ug, attrs={"markers": {"structured": {}}})
        with pytest.raises(NotImplementedError):
            write_su2("/tmp/bad.su2", ds)


# =====================================================================
#  Duplicate marker merge test
# =====================================================================

class TestDuplicateMarkers:
    """Two sides mapping to the same tag should merge into one block."""

    def test_merged_boundary(self):
        grid = _make_grid_3x2()
        # Both imin and imax map to "farfield"
        markers = {
            "imin": "farfield",
            "imax": "farfield",
            "jmin": "wall",
            "jmax": "top",
        }
        _, _, boundaries = structured_to_unstructured(grid, markers)

        # "farfield" should appear once with concatenated segments
        assert "farfield" in boundaries
        # imin has 1 segment, imax has 1 segment → merged = 2 segments
        assert boundaries["farfield"].shape == (2, 2)

    def test_merged_block_position(self):
        """Merged marker appears at the position of the first contributing side."""
        grid = _make_grid_3x2()
        markers = {
            "imin": "farfield",
            "imax": "farfield",
            "jmin": "wall",
            "jmax": "top",
        }
        _, _, boundaries = structured_to_unstructured(grid, markers)
        # "farfield" comes from imin (first in canonical order)
        # so it should be the first key
        assert list(boundaries.keys())[0] == "farfield"


# =====================================================================
#  3-D smoke test
# =====================================================================

class TestConversion3D:
    """Basic 3-D conversion checks on a 3×3×3 grid."""

    def setup_method(self):
        grid = _make_grid_3d(3, 3, 3)
        markers = _make_markers_3d()
        self.points, self.elements, self.boundaries = (
            structured_to_unstructured(grid, markers)
        )
        self.ni, self.nj, self.nk = 3, 3, 3

    def test_point_count(self):
        assert self.points.shape == (27, 3)

    def test_element_count(self):
        # (3-1)*(3-1)*(3-1) = 8 hexes
        assert self.elements.shape == (8, 8)

    def test_all_node_ids_valid(self):
        assert np.all(self.elements >= 0)
        assert np.all(self.elements < 27)

    def test_boundary_face_counts(self):
        # Each face of a 3×3×3 grid has (3-1)*(3-1)=4 quads
        total_bfaces = sum(b.shape[0] for b in self.boundaries.values())
        assert total_bfaces == 6 * 4  # 6 sides × 4 faces each

    def test_adjacent_hexes_share_four_nodes(self):
        """Adjacent hexes should share exactly 4 nodes (one face)."""
        # Pick first two hexes and verify they share a face
        h0 = set(self.elements[0])
        h1 = set(self.elements[1])
        shared = h0 & h1
        assert len(shared) == 4

    def test_boundary_faces_are_volume_faces(self):
        """Every boundary quad must be a face of some volume hex."""
        # Build set of all hex faces
        hex_faces = set()
        for elem in self.elements:
            # The 6 faces of a hex (using SU2 node ordering)
            faces = [
                frozenset([elem[0], elem[1], elem[2], elem[3]]),  # bottom
                frozenset([elem[4], elem[5], elem[6], elem[7]]),  # top
                frozenset([elem[0], elem[1], elem[5], elem[4]]),  # front
                frozenset([elem[2], elem[3], elem[7], elem[6]]),  # back
                frozenset([elem[0], elem[3], elem[7], elem[4]]),  # left
                frozenset([elem[1], elem[2], elem[6], elem[5]]),  # right
            ]
            hex_faces.update(faces)

        for tag, bfaces in self.boundaries.items():
            for face in bfaces:
                fset = frozenset(int(n) for n in face)
                assert fset in hex_faces, (
                    f"boundary face {face} (marker={tag}) is not "
                    f"a face of any volume hex"
                )


# =====================================================================
#  Serializer tests
# =====================================================================

class TestSerializer:
    """Verify the SU2 file format output."""

    def test_file_sections(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()
        lines = text.splitlines()

        # Check section headers
        assert lines[0] == "NDIME= 2"
        assert lines[1] == "NELEM= 2"
        # 2 element lines, then NPOIN
        assert lines[4] == "NPOIN= 6"
        # 6 point lines, then NMARK
        assert lines[11] == "NMARK= 4"

    def test_element_format(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()
        lines = text.splitlines()

        # First element: "9 0 1 4 3 0"
        assert lines[2] == "9 0 1 4 3 0"
        # Second element: "9 1 2 5 4 1"
        assert lines[3] == "9 1 2 5 4 1"

    def test_coordinate_format(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()
        lines = text.splitlines()

        # First point: node 0 at (0.0, 0.0)
        point_line = lines[5]
        parts = point_line.split()
        assert len(parts) == 3  # x, y, node_id
        assert parts[2] == "0"
        # Verify scientific notation
        assert "e" in parts[0].lower()
        assert "e" in parts[1].lower()

    def test_marker_order(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()

        # Markers should appear in canonical side order
        tags = []
        for line in text.splitlines():
            if line.startswith("MARKER_TAG="):
                tags.append(line.split("= ")[1])
        assert tags == ["inlet", "outlet", "wall", "farfield"]

    def test_boundary_element_format(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()
        lines = text.splitlines()

        # Find the wall marker and check line element format
        for i, line in enumerate(lines):
            if line == "MARKER_TAG= wall":
                assert lines[i + 1] == "MARKER_ELEMS= 2"
                assert lines[i + 2] == "3 0 1"
                assert lines[i + 3] == "3 1 2"
                break
        else:
            pytest.fail("MARKER_TAG= wall not found in output")

    def test_zero_based_indexing(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        text = out.read_text()

        # All node IDs in element lines should be 0-based
        for line in text.splitlines():
            if line.startswith("9 "):  # quad element
                parts = line.split()
                node_ids = [int(p) for p in parts[1:-1]]
                for nid in node_ids:
                    assert 0 <= nid < 6
            elif line.startswith("3 "):  # line element
                parts = line.split()
                node_ids = [int(p) for p in parts[1:]]
                for nid in node_ids:
                    assert 0 <= nid < 6

    def test_returns_path(self, tmp_path):
        ds = _make_dataset_3x2()
        out = write_su2(tmp_path / "mesh.su2", ds)
        assert isinstance(out, Path)
        assert out.exists()


# =====================================================================
#  Dispatch integration test
# =====================================================================

class TestDispatch:
    """Verify write_file() dispatches .su2 to the SU2 writer."""

    def test_write_file_su2(self, tmp_path):
        from cfd_io.convert_mod import write_file

        ds = _make_dataset_3x2()
        out = write_file(tmp_path / "mesh.su2", ds)
        assert out.exists()
        text = out.read_text()
        assert text.startswith("NDIME= 2")
