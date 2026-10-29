"""Tests for the Gmsh (.msh, MSH 4.1 ASCII) writer.

Covers:
- File structure parsed back by a small reader (always runs)
- Second-order elements: node counts, boundary lines matching quad faces
- Orientation: counter-clockwise quads from clockwise grids, folded grids rejected
- Negative tests (markers, point counts, reserved name)
- Read-back with the gmsh Python package (skipped when it isn't available)
"""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.convert_mod import write_file
from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.writers.gmsh import write_gmsh

# -- shared test data --------------------------------------------------
MARKERS = {"imin": "symmetry", "imax": "outflow", "jmin": "wall", "jmax": "farfield"}

# quad corner positions in the Gmsh node order (corners come first at both orders)
N_CORNERS = 4


def _annulus(ni: int, nj: int, clockwise: bool = False) -> Dataset:
    """Quarter annulus (radius 1 to 2): i around the arc, j outward. Curved, so order 2 matters."""
    theta = np.linspace(np.pi, np.pi / 2, ni)
    if clockwise:
        theta = theta[::-1]
    radius = np.linspace(1.0, 2.0, nj)
    x = radius[None, :] * np.cos(theta[:, None])
    y = radius[None, :] * np.sin(theta[:, None])
    grid = StructuredGrid(x[:, :, None], y[:, :, None], np.zeros((ni, nj, 1)))
    return Dataset(grid=grid, attrs={"markers": {"structured": dict(MARKERS)}})


def _read_msh(path: Path) -> dict:
    """Parse the sections this writer produces into plain Python data."""

    # split into lines
    lines = path.read_text().splitlines()

    # physical names: tag -> (dimension, name)
    start = lines.index("$PhysicalNames")
    n_names = int(lines[start + 1])
    physical = {}
    for line in lines[start + 2 : start + 2 + n_names]:
        dim, tag, name = line.split(maxsplit=2)
        physical[int(tag)] = (int(dim), name.strip('"'))

    # entities: (dimension, tag) -> physical tag
    start = lines.index("$Entities")
    n_points, n_curves, n_surfaces, _ = (int(v) for v in lines[start + 1].split())
    entity_physical = {}
    for offset in range(n_curves):
        fields = lines[start + 2 + n_points + offset].split()
        entity_physical[(1, int(fields[0]))] = int(fields[8])
    for offset in range(n_surfaces):
        fields = lines[start + 2 + n_points + n_curves + offset].split()
        entity_physical[(2, int(fields[0]))] = int(fields[8])

    # nodes: one block, tags then coordinates
    start = lines.index("$Nodes")
    n_nodes = int(lines[start + 1].split()[1])
    tags = np.array([int(v) for v in lines[start + 3 : start + 3 + n_nodes]])
    coords = np.array(
        [[float(v) for v in line.split()] for line in lines[start + 3 + n_nodes : start + 3 + 2 * n_nodes]]
    )

    # elements: blocks of (dimension, entity, type) -> node tag rows
    start = lines.index("$Elements")
    n_blocks = int(lines[start + 1].split()[0])
    blocks = []
    cursor = start + 2
    for _ in range(n_blocks):
        dim, entity, etype, count = (int(v) for v in lines[cursor].split())
        rows = np.array([[int(v) for v in line.split()] for line in lines[cursor + 1 : cursor + 1 + count]])
        blocks.append({"dim": dim, "entity": entity, "type": etype, "nodes": rows[:, 1:]})
        cursor += 1 + count

    return {
        "physical": physical,
        "entity_physical": entity_physical,
        "node_tags": tags,
        "coords": coords,
        "blocks": blocks,
    }


def _signed_corner_areas(mesh: dict) -> np.ndarray:
    """Signed area of each quad from its corner nodes (positive = counter-clockwise)."""
    quads = next(b for b in mesh["blocks"] if b["dim"] == 2)["nodes"]
    corners = mesh["coords"][quads[:, :N_CORNERS] - 1]
    x, y = corners[..., 0], corners[..., 1]
    return 0.5 * ((x[:, 2] - x[:, 0]) * (y[:, 3] - y[:, 1]) - (x[:, 3] - x[:, 1]) * (y[:, 2] - y[:, 0]))


# =====================================================================
#  File structure
# =====================================================================

@pytest.mark.parametrize("order", [1, 2])
def test_nodes_elements_and_groups(tmp_path: Path, order: int) -> None:
    """Every grid point is a node, element counts and types match the order, groups are named."""

    # write a 5 x 3 grid
    path = tmp_path / "grid.msh"
    dataset = _annulus(5, 3)
    write_gmsh(path, dataset, order=order)
    mesh = _read_msh(path)

    # check nodes: tag 1 + i + ni * j holds point (i, j)
    np.testing.assert_array_equal(mesh["node_tags"], np.arange(1, 16))
    np.testing.assert_array_equal(mesh["coords"][:, 0], dataset.grid.x.ravel(order="F"))
    np.testing.assert_array_equal(mesh["coords"][:, 1], dataset.grid.y.ravel(order="F"))

    # check elements: types (line, quad) and counts per order
    expected = {1: {1: 12, 3: 8}, 2: {8: 6, 10: 2}}[order]
    counts: dict[int, int] = {}
    for block in mesh["blocks"]:
        counts[block["type"]] = counts.get(block["type"], 0) + block["nodes"].shape[0]
    assert counts == expected

    # check groups: boundaries 1..4 (dimension 1), fluid 5 (dimension 2), tags unique
    assert mesh["physical"] == {
        1: (1, "symmetry"),
        2: (1, "outflow"),
        3: (1, "wall"),
        4: (1, "farfield"),
        5: (2, "fluid"),
    }


def test_sides_sharing_a_name_share_a_group(tmp_path: Path) -> None:
    """Two sides with the same marker name end up in one physical group."""

    # imin and imax both "farfield"
    dataset = _annulus(5, 3)
    dataset.attrs["markers"]["structured"]["imax"] = "farfield"
    path = tmp_path / "grid.msh"
    write_gmsh(path, dataset)
    mesh = _read_msh(path)

    # check: three boundary groups plus fluid; both i-side curves point at farfield's tag
    names = [name for _, name in mesh["physical"].values()]
    assert names == ["symmetry", "farfield", "wall", "fluid"]
    assert mesh["entity_physical"][(1, 2)] == mesh["entity_physical"][(1, 4)]


def test_second_order_boundary_lines_match_quad_faces(tmp_path: Path) -> None:
    """Each 3-node boundary line has exactly the nodes of one quad face (boundaries are matched to faces by their nodes)."""

    # write
    path = tmp_path / "grid.msh"
    write_gmsh(path, _annulus(7, 5), order=2)
    mesh = _read_msh(path)

    # faces of the 9-node quads: (corner, corner, edge midpoint), Gmsh node order
    quads = next(b for b in mesh["blocks"] if b["dim"] == 2)["nodes"]
    face_nodes = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
    quad_faces = set()
    for face in face_nodes:
        for row in quads[:, face]:
            quad_faces.add((frozenset(row[:2]), row[2]))

    # check: every boundary line is a quad face, with its middle node in the middle
    line_blocks = [b for b in mesh["blocks"] if b["dim"] == 1]
    for block in line_blocks:
        assert block["nodes"].shape[1] == 3
        for row in block["nodes"]:
            assert (frozenset(row[:2]), row[2]) in quad_faces


# =====================================================================
#  Orientation
# =====================================================================

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("clockwise", [False, True])
def test_quads_are_counter_clockwise(tmp_path: Path, order: int, clockwise: bool) -> None:
    """Quads come out counter-clockwise in x-y, also from a grid that is clockwise in (i, j)."""

    # write
    path = tmp_path / "grid.msh"
    write_gmsh(path, _annulus(5, 3, clockwise=clockwise), order=order)

    # check
    assert np.all(_signed_corner_areas(_read_msh(path)) > 0.0)


def test_folded_grid_raises(tmp_path: Path) -> None:
    """A grid with cells of both orientations is rejected and no file is written."""

    # fold the grid: swap two i-columns
    dataset = _annulus(5, 3)
    dataset.grid.x[[1, 2]] = dataset.grid.x[[2, 1]]
    dataset.grid.y[[1, 2]] = dataset.grid.y[[2, 1]]

    # check
    path = tmp_path / "grid.msh"
    with pytest.raises(ValueError, match="folded"):
        write_gmsh(path, dataset)
    assert not path.exists()


# =====================================================================
#  Negative tests
# =====================================================================

@pytest.mark.parametrize("shape", [(4, 3), (5, 4)])
def test_second_order_needs_odd_point_counts(tmp_path: Path, shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="odd"):
        write_gmsh(tmp_path / "grid.msh", _annulus(*shape), order=2)


def test_three_dimensional_grid_rejected(tmp_path: Path) -> None:
    grid = StructuredGrid(np.zeros((3, 3, 2)), np.zeros((3, 3, 2)), np.zeros((3, 3, 2)))
    with pytest.raises(ValueError, match=r"\(ni, nj, 1\)"):
        write_gmsh(tmp_path / "grid.msh", Dataset(grid=grid, attrs={}))


def test_missing_markers_raise(tmp_path: Path) -> None:
    dataset = _annulus(5, 3)
    del dataset.attrs["markers"]["structured"]["jmax"]
    with pytest.raises(ValueError, match="missing required marker keys"):
        write_gmsh(tmp_path / "grid.msh", dataset)


def test_fluid_is_reserved(tmp_path: Path) -> None:
    dataset = _annulus(5, 3)
    dataset.attrs["markers"]["structured"]["jmax"] = "Fluid"
    with pytest.raises(ValueError, match="reserved"):
        write_gmsh(tmp_path / "grid.msh", dataset)


def test_write_file_dispatches_msh(tmp_path: Path) -> None:
    """The .msh extension goes to the Gmsh writer (first order)."""
    path = write_file(tmp_path / "grid.msh", _annulus(5, 3))
    assert path.read_text().startswith("$MeshFormat\n4.1 0 8\n")


# =====================================================================
#  Read-back with the gmsh package
# =====================================================================

@pytest.mark.parametrize("order", [1, 2])
def test_gmsh_reads_it_back(tmp_path: Path, order: int) -> None:
    """Gmsh itself reads the file: nodes, element types, physical groups, positive Jacobians."""

    # gmsh is optional; its wheel can also fail to load without system graphics libraries
    try:
        import gmsh
    except (ImportError, OSError) as error:
        pytest.skip(f"gmsh not available: {error}")

    # write
    path = tmp_path / "grid.msh"
    write_gmsh(path, _annulus(5, 3), order=order)

    # read
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        groups = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups()}
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements()
        quad_type = {1: 3, 2: 10}[order]
        quad_tags = element_tags[list(element_types).index(quad_type)]
        min_jacobian = gmsh.model.mesh.getElementQualities(quad_tags, "minSJ").min()
    finally:
        gmsh.finalize()

    # check
    assert groups == {"symmetry", "outflow", "wall", "farfield", "fluid"}
    assert len(node_tags) == 15
    assert min_jacobian > 0.0
