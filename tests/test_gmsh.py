"""Tests for the Gmsh (.msh, MSH 4.1 ASCII) writer.

Covers:
- File structure parsed back by a small reader (always runs)
- Second-order elements: node counts, boundary lines matching quad faces
- Orientation: counter-clockwise quads from clockwise grids, folded grids rejected
- Negative tests (markers, point counts, reserved name)
- 3-D: hexahedra, the axis of a revolved grid (merged nodes, prisms, triangles,
  no group for the axis face), orientation, collapsed faces
- Read-back with the gmsh Python package (skipped when it isn't available)
"""

from pathlib import Path

import numpy as np
import pytest
from revolved import MARKERS_3D, revolved_sector

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
    n_points, n_curves, n_surfaces, n_volumes = (int(v) for v in lines[start + 1].split())
    entity_physical = {}
    for offset in range(n_curves):
        fields = lines[start + 2 + n_points + offset].split()
        entity_physical[(1, int(fields[0]))] = int(fields[8])
    for offset in range(n_surfaces):
        fields = lines[start + 2 + n_points + n_curves + offset].split()
        entity_physical[(2, int(fields[0]))] = int(fields[8])
    for offset in range(n_volumes):
        fields = lines[start + 2 + n_points + n_curves + n_surfaces + offset].split()
        entity_physical[(3, int(fields[0]))] = int(fields[8])

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
#  3-D grids
# =====================================================================

# Gmsh element types: (quad, triangle, hexahedron, prism) per order
TYPES_3D = {1: (3, 2, 5, 6), 2: (10, 9, 12, 13)}


def _box(ni: int, nj: int, nk: int) -> Dataset:
    """A unit box, no axis."""
    x, y, z = np.meshgrid(
        np.linspace(0, 1, ni), np.linspace(0, 1, nj), np.linspace(0, 1, nk), indexing="ij"
    )
    grid = StructuredGrid(x, y, z)
    return Dataset(grid=grid, attrs={"markers": {"structured": dict(MARKERS_3D)}})


def _element_counts(mesh: dict) -> dict[int, int]:
    """Number of elements per Gmsh element type."""
    counts: dict[int, int] = {}
    for block in mesh["blocks"]:
        counts[block["type"]] = counts.get(block["type"], 0) + block["nodes"].shape[0]
    return counts


def _volume_corners_signed(mesh: dict) -> np.ndarray:
    """Signed triple product at the first corner of every hexahedron and prism.

    Hexahedron: edges to corners 1, 3, 4; prism: edges to corners 1, 2, 3. Both are
    positive for Gmsh's right-handed node order.
    """
    coords = mesh["coords"]
    products = []
    for block in mesh["blocks"]:
        if block["dim"] != 3:
            continue
        nodes = block["nodes"] - 1
        if block["type"] in (5, 12):
            corner_ids = [0, 1, 3, 4]
        else:
            corner_ids = [0, 1, 2, 3]
        p0, p1, p2, p3 = (coords[nodes[:, c]] for c in corner_ids)
        products.append(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0))
    return np.concatenate(products)


@pytest.mark.parametrize("order", [1, 2])
def test_box_becomes_hexahedra_and_quads(tmp_path: Path, order: int) -> None:
    """Without an axis: every grid point a node, one hexahedron per cell (block), six side groups."""

    # write a 5 x 3 x 3 box
    path = tmp_path / "box.msh"
    write_gmsh(path, _box(5, 3, 3), order=order)
    mesh = _read_msh(path)

    # check: 45 nodes, hexahedra and quads per order, no triangles or prisms
    quad, _, hexahedron, _ = TYPES_3D[order]
    expected = {1: {quad: 2 * (8 + 8 + 4), hexahedron: 16}, 2: {quad: 2 * (2 + 2 + 1), hexahedron: 2}}
    assert mesh["coords"].shape == (45, 3)
    assert _element_counts(mesh) == expected[order]

    # check: boundaries of dimension 2, fluid of dimension 3
    assert sorted(mesh["physical"].values()) == sorted(
        [(2, name) for name in MARKERS_3D.values()] + [(3, "fluid")]
    )


@pytest.mark.parametrize("order", [1, 2])
def test_axis_is_merged_into_prisms(tmp_path: Path, order: int) -> None:
    """A revolved grid: one node per axis line, prisms and triangles at the axis, no axis group."""

    # write a 9 x 5 x 7 sector
    ni, nj, nk = 9, 5, 7
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(ni, nj, nk), order=order)
    mesh = _read_msh(path)

    # check nodes: the axis lines' duplicates merged; at order 2 also the points
    # inside the axis prisms (i = 1, odd k) left out, since no element uses them
    n_nodes = ni * nj * nk - nj * (nk - 1)
    if order == 2:
        n_nodes -= nj * (nk - 1) // 2
    assert mesh["coords"].shape == (n_nodes, 3)

    # check elements: prisms in the first i-layer, triangles where wall and farfield
    # touch the axis
    o = order
    cells_i, cells_j, cells_k = (ni - 1) // o, (nj - 1) // o, (nk - 1) // o
    quad, triangle, hexahedron, prism = TYPES_3D[order]
    counts = _element_counts(mesh)
    assert counts[prism] == cells_j * cells_k
    assert counts[hexahedron] == (cells_i - 1) * cells_j * cells_k
    assert counts[triangle] == 2 * cells_k

    # check groups: no axis group (the face has no area)
    names = {name for _, name in mesh["physical"].values()}
    assert names == {"outflow", "wall", "farfield", "sweep_low", "sweep_high", "fluid"}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("left_handed", [False, True])
def test_volume_elements_are_right_handed(tmp_path: Path, order: int, left_handed: bool) -> None:
    """Hexahedra and prisms come out right-handed, also from a left-handed grid."""

    # write
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(9, 5, 7, left_handed=left_handed), order=order)

    # check
    assert np.all(_volume_corners_signed(_read_msh(path)) > 0.0)


def test_left_handed_grid_keeps_its_names_on_its_planes(tmp_path: Path) -> None:
    """Reversing k for a left-handed grid leaves kmin's name on the grid's k = 0 plane."""

    # left-handed: k = 0 is the plane at 10 degrees
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(9, 5, 7, left_handed=True))
    mesh = _read_msh(path)

    # nodes of the sweep_low faces
    tag = next(t for t, (_, name) in mesh["physical"].items() if name == "sweep_low")
    surface = next(e for (d, e), t in mesh["entity_physical"].items() if d == 2 and t == tag)
    nodes = np.concatenate([b["nodes"].ravel() for b in mesh["blocks"] if b["dim"] == 2 and b["entity"] == surface])
    theta = np.arctan2(mesh["coords"][nodes - 1, 2], mesh["coords"][nodes - 1, 1])

    # check: on the 10-degree plane (or the axis)
    on_axis = np.hypot(mesh["coords"][nodes - 1, 1], mesh["coords"][nodes - 1, 2]) == 0.0
    np.testing.assert_allclose(theta[~on_axis], np.radians(10.0))


def test_boundary_faces_are_element_faces(tmp_path: Path) -> None:
    """Every boundary quad and triangle has the corners of one face of a volume element."""

    # write
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(9, 5, 7))
    mesh = _read_msh(path)

    # faces of the volume elements, as corner sets (Gmsh corner order)
    hexahedron_faces = [(0, 3, 2, 1), (0, 1, 5, 4), (0, 4, 7, 3), (1, 2, 6, 5), (2, 3, 7, 6), (4, 5, 6, 7)]
    prism_faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (0, 3, 5, 2), (1, 2, 5, 4)]
    element_faces = set()
    for block in mesh["blocks"]:
        if block["dim"] != 3:
            continue
        faces = hexahedron_faces if block["type"] == 5 else prism_faces
        for row in block["nodes"]:
            for face in faces:
                element_faces.add(frozenset(row[list(face)]))

    # check
    for block in mesh["blocks"]:
        if block["dim"] == 2:
            for row in block["nodes"]:
                assert frozenset(row) in element_faces


def test_second_order_nodes_on_the_arc(tmp_path: Path) -> None:
    """Order 2: the midpoint of each hexahedron edge along k lies on the arc, at the mid-angle."""

    # write
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(9, 5, 7), order=2)
    mesh = _read_msh(path)

    # hexahedron edge 0-4 runs along k; its midpoint is node 10
    hexahedra = next(b for b in mesh["blocks"] if b["type"] == 12)["nodes"] - 1
    coords = mesh["coords"]
    start, end, middle = coords[hexahedra[:, 0]], coords[hexahedra[:, 4]], coords[hexahedra[:, 10]]

    # check: same radius, angle halfway
    def radius(p):
        return np.hypot(p[:, 1], p[:, 2])

    def angle(p):
        return np.arctan2(p[:, 2], p[:, 1])

    np.testing.assert_allclose(radius(middle), radius(start))
    np.testing.assert_allclose(angle(middle), 0.5 * (angle(start) + angle(end)))


def test_folded_3d_grid_raises(tmp_path: Path) -> None:
    """A 3-D grid with cells of both orientations (or none) is refused."""
    dataset = _box(4, 3, 3)
    dataset.grid.x[[1, 2]] = dataset.grid.x[[2, 1]]
    with pytest.raises(ValueError, match="folded"):
        write_gmsh(tmp_path / "grid.msh", dataset)


def test_collapse_other_than_imin_raises(tmp_path: Path) -> None:
    """Only the face i = 0 may lie on an axis."""
    dataset = revolved_sector(9, 5, 7)
    grid = dataset.grid
    flipped = StructuredGrid(grid.x[::-1].copy(), grid.y[::-1].copy(), grid.z[::-1].copy())
    # (reversing i makes the grid left-handed in i too; reverse k back to keep it valid)
    flipped = StructuredGrid(flipped.x[:, :, ::-1].copy(), flipped.y[:, :, ::-1].copy(), flipped.z[:, :, ::-1].copy())
    with pytest.raises(ValueError, match="imax face collapses"):
        write_gmsh(tmp_path / "grid.msh", Dataset(grid=flipped, attrs=dataset.attrs))


def test_second_order_3d_needs_odd_point_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="odd"):
        write_gmsh(tmp_path / "grid.msh", revolved_sector(9, 5, 6), order=2)


def test_3d_needs_six_markers(tmp_path: Path) -> None:
    dataset = _box(3, 3, 3)
    del dataset.attrs["markers"]["structured"]["kmax"]
    with pytest.raises(ValueError, match="missing required marker keys"):
        write_gmsh(tmp_path / "grid.msh", dataset)


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


@pytest.mark.parametrize("order", [1, 2])
def test_gmsh_reads_a_revolved_grid_back(tmp_path: Path, order: int) -> None:
    """Gmsh reads the sector: groups, and positive Jacobians of hexahedra and prisms."""

    # gmsh is optional; its wheel can also fail to load without system graphics libraries
    try:
        import gmsh
    except (ImportError, OSError) as error:
        pytest.skip(f"gmsh not available: {error}")

    # write
    path = tmp_path / "sector.msh"
    write_gmsh(path, revolved_sector(9, 5, 7), order=order)

    # read
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        groups = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups()}
        element_types, element_tags, _ = gmsh.model.mesh.getElements(dim=3)
        min_jacobian = {}
        for element_type, tags in zip(element_types, element_tags):
            min_jacobian[int(element_type)] = gmsh.model.mesh.getElementQualities(tags, "minSJ").min()
    finally:
        gmsh.finalize()

    # check
    _, _, hexahedron, prism = TYPES_3D[order]
    assert groups == {"outflow", "wall", "farfield", "sweep_low", "sweep_high", "fluid"}
    assert set(min_jacobian) == {hexahedron, prism}
    assert min(min_jacobian.values()) > 0.0
