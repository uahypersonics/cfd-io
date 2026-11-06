"""Volume mesh of a 3-D structured grid, shared by the SU2 and Gmsh writers.

Each grid cell (each 2 x 2 x 2 block of cells at order 2) becomes a hexahedron, and
each cell face on a grid side a quadrilateral. Element nodes are listed as
``(di, dj, dk)`` offsets from the element's first grid point, in Gmsh's node order;
the SU2 writer reorders the prisms for VTK.

**Axis.** A grid revolved about an axis has its face ``i = 0`` on the axis: each of
its k-lines collapses to a single point. Those points are merged into one node per
line, the cells touching the axis become prisms (their two edges on the axis are one
edge each), the side faces touching the axis become triangles, and the face
``i = 0`` itself, which has no area, gets no faces: the axis is an edge inside the
domain, or between the two side faces of a sector. Only this case is supported; a
grid with any other collapsed face is refused.

**Orientation.** Elements are written right-handed (positive volume). A grid that is
left-handed in ``(i, j, k)`` is turned right-handed by reversing k (the markers of
``kmin`` and ``kmax`` swap with it); a grid with cells of both orientations, or of no
volume, is refused.

Nodes keep the grid's order (i fastest, then j, then k), without the merged axis
points and without grid points that no element uses (at order 2, the points inside
the axis prisms).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cfd_io.dataset import StructuredGrid

# --------------------------------------------------
# constants
# --------------------------------------------------

# a k-line is collapsed when its length is below this fraction of the next line's
# length (the next line away from the face); revolved grids give exactly zero
COLLAPSE_TOLERANCE = 1.0e-9

# hexahedron nodes as (di, dj, dk) offsets, Gmsh order: the 8 corners (face k = 0
# counter-clockwise in (i, j), then face k = 1), at order 2 then the 12 edge
# midpoints, the 6 face centres and the centre
_HEX_OFFSETS = {
    1: [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
    2: [
        # corners
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
        (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
        # edge midpoints
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 0),
        (2, 0, 1), (1, 2, 0), (2, 2, 1), (0, 2, 1),
        (1, 0, 2), (0, 1, 2), (2, 1, 2), (1, 2, 2),
        # face centres, then the centre
        (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1),
        (1, 2, 1), (1, 1, 2), (1, 1, 1),
    ],
}

# prism nodes of a cell touching the axis (di = 0 is the axis, collapsed along dk),
# Gmsh order: triangle (axis, (o, o), (o, 0)) at dj = 0, the same at dj = o, then at
# order 2 the edge midpoints and the centres of the three quadrilateral faces. The
# triangles are the cell's faces of constant j, and Gmsh wants the first one's normal
# (right-hand rule) to point at the second: in (i, j, k), from the axis to dk = o
# first, then to dk = 0. The grid points (1, dj, 1) inside the prism aren't nodes.
_PRISM_OFFSETS = {
    1: [
        (0, 0, 0), (1, 0, 1), (1, 0, 0),
        (0, 1, 0), (1, 1, 1), (1, 1, 0),
    ],
    2: [
        # corners
        (0, 0, 0), (2, 0, 2), (2, 0, 0),
        (0, 2, 0), (2, 2, 2), (2, 2, 0),
        # edge midpoints, in Gmsh's edge order 0-1, 0-2, 0-3, 1-2, 1-4, 2-5, 3-4, 3-5, 4-5
        (1, 0, 2), (1, 0, 0), (0, 1, 0),
        (2, 0, 1), (2, 1, 2), (2, 1, 0),
        (1, 2, 2), (1, 2, 0), (2, 2, 1),
        # centres of the quadrilateral faces
        (1, 1, 2), (1, 1, 0), (2, 1, 1),
    ],
}

# quadrilateral face nodes as (da, db) offsets in the side's own (a, b) frame, Gmsh
# order: corners counter-clockwise about the outward normal, then at order 2 the edge
# midpoints and the centre
_QUAD_OFFSETS = {
    1: [(0, 0), (1, 0), (1, 1), (0, 1)],
    2: [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)],
}

# triangle face nodes of a face touching the axis: the quadrilateral without its
# collapsed edge (the corners keep their turn about the outward normal), keyed by
# which in-plane direction runs along i (the collapsed edge is where it is 0)
_AXIS_TRIANGLE_OFFSETS = {
    # a along i: corners (0, 0) and (0, o) merge
    "a": {
        1: [(0, 0), (1, 0), (1, 1)],
        2: [(0, 0), (2, 0), (2, 2), (1, 0), (2, 1), (1, 2)],
    },
    # b along i: corners (0, 0) and (o, 0) merge
    "b": {
        1: [(1, 0), (1, 1), (0, 1)],
        2: [(2, 0), (2, 2), (0, 2), (2, 1), (1, 2), (0, 1)],
    },
}

# each grid side: its fixed axis and index ("first" or "last"), and its in-plane
# axes (a, b), chosen so that a x b points out of the grid (axes 0, 1, 2 = i, j, k)
_SIDE_FRAMES = {
    "imin": (0, "first", 2, 1),
    "imax": (0, "last", 1, 2),
    "jmin": (1, "first", 0, 2),
    "jmax": (1, "last", 2, 0),
    "kmin": (2, "first", 1, 0),
    "kmax": (2, "last", 0, 1),
}

# sides in the order they are written
SIDES_3D = ["imin", "imax", "jmin", "jmax", "kmin", "kmax"]


# --------------------------------------------------
# result
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class VolumeMesh:
    """Nodes, volume elements and side faces of a 3-D structured grid.

    Attributes:
        points: Node coordinates, shape (n_nodes, 3).
        hexahedra: Node ids (0-based) per hexahedron, shape (n, 8 or 27), Gmsh order.
        prisms: Node ids per prism at the axis, shape (n, 6 or 18), Gmsh order;
            empty without an axis.
        faces: Per side, the node ids of its quadrilaterals and triangles:
            faces[side]["quadrilateral"] (n, 4 or 9) and faces[side]["triangle"]
            (n, 3 or 6). A side on the axis has none of either.
        axis: True when the face i = 0 is an axis (merged nodes, prisms).
        k_reversed: True when the grid was left-handed and k was reversed: the
            kmin faces lie on the grid's last k-plane, kmax on its first.
    """

    points: NDArray[np.float64]
    hexahedra: NDArray[np.int64]
    prisms: NDArray[np.int64]
    faces: dict[str, dict[str, NDArray[np.int64]]]
    axis: bool
    k_reversed: bool


# --------------------------------------------------
# public API (inside cfd-io)
# --------------------------------------------------
def build_volume_mesh(grid: StructuredGrid, order: int) -> VolumeMesh:
    """Turn a 3-D structured grid into hexahedra (prisms at an axis) and side faces.

    Args:
        grid: Structured grid of shape (ni, nj, nk), all >= 2.
        order: Element order: 1, or 2 (elements span 2 grid cells per direction,
            so ni, nj and nk must be odd).

    Returns:
        The volume mesh.

    Raises:
        ValueError: If the grid is folded or has cells of no volume, has a collapsed
            face other than an axis at i = 0, or its point counts don't suit order.
    """

    # validate inputs
    ni, nj, nk = grid.shape
    if min(ni, nj, nk) < 2:
        raise ValueError(f"a 3-D volume mesh needs ni, nj, nk >= 2, got {grid.shape}")
    if order == 2 and (ni % 2 == 0 or nj % 2 == 0 or nk % 2 == 0):
        raise ValueError(f"order 2 needs odd ni, nj and nk, got {ni}, {nj}, {nk}")

    # coordinates as (ni, nj, nk, 3)
    xyz = np.stack([grid.x, grid.y, grid.z], axis=-1).astype(np.float64)

    # check orientation: right-handed, or reverse k to make it so
    k_reversed = _is_left_handed(xyz)
    if k_reversed:
        xyz = xyz[:, :, ::-1, :]

    # check for an axis at i = 0 (and refuse other collapsed faces)
    axis = _find_axis(xyz)

    # structured index of every grid point; axis points all name their line's k = 0 point
    index = np.arange(ni * nj * nk).reshape((ni, nj, nk), order="F")
    if axis:
        index[0, :, :] = index[0, :, :1]

    # first grid point of each element along each direction
    i_start = np.arange(0, ni - 1, order)
    j_start = np.arange(0, nj - 1, order)
    k_start = np.arange(0, nk - 1, order)

    # volume elements: prisms in the first i-layer when there is an axis, hexahedra elsewhere
    hex_i_start = i_start
    prisms = np.empty((0, len(_PRISM_OFFSETS[order])), dtype=np.int64)
    if axis:
        prisms = _element_nodes(index, i_start[:1], j_start, k_start, _PRISM_OFFSETS[order])
        hex_i_start = i_start[1:]
    hexahedra = _element_nodes(index, hex_i_start, j_start, k_start, _HEX_OFFSETS[order])

    # side faces
    faces = {}
    for side in SIDES_3D:
        faces[side] = _side_faces(index, side, order, axis)

    # compact the node numbering: only the points the elements use, in grid order
    used = np.unique(np.concatenate([hexahedra.ravel(), prisms.ravel()]))
    points = xyz.reshape((-1, 3), order="F")[used]
    hexahedra = np.searchsorted(used, hexahedra)
    prisms = np.searchsorted(used, prisms)
    for side_faces in faces.values():
        for shape_name, nodes in side_faces.items():
            side_faces[shape_name] = np.searchsorted(used, nodes)

    mesh = VolumeMesh(
        points=points,
        hexahedra=hexahedra,
        prisms=prisms,
        faces=faces,
        axis=axis,
        k_reversed=k_reversed,
    )
    return mesh


# --------------------------------------------------
# helpers: checks
# --------------------------------------------------
def _cell_volumes(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Volume of every grid cell, from six tetrahedra around its diagonal (0, 0, 0)-(1, 1, 1).

    The tetrahedra share the diagonal and fill a cell with planar faces exactly; a cell
    touching an axis (two corners merged with two others) gets its prism volume, since
    the two tetrahedra with merged corners have none. Right-handed cells are positive.
    """

    # cell corners, (ni - 1, nj - 1, nk - 1, 3) each, named by (di, dj, dk)
    def corner(di: int, dj: int, dk: int) -> NDArray[np.float64]:
        ni, nj, nk, _ = xyz.shape
        return xyz[di : ni - 1 + di, dj : nj - 1 + dj, dk : nk - 1 + dk]

    # the diagonal, and the ring of six corners around it (each step turns about it
    # the same way)
    start = corner(0, 0, 0)
    end = corner(1, 1, 1)
    ring = [
        corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 0),
        corner(0, 1, 1), corner(0, 0, 1), corner(1, 0, 1),
    ]

    # six tetrahedra (start, ring[n], ring[n + 1], end)
    volumes = np.zeros(start.shape[:3])
    diagonal = end - start
    for n in range(6):
        edge_1 = ring[n] - start
        edge_2 = ring[(n + 1) % 6] - start
        triple = np.einsum("...i,...i->...", np.cross(edge_1, edge_2), diagonal)
        volumes = volumes + triple / 6.0

    return volumes


def _is_left_handed(xyz: NDArray[np.float64]) -> bool:
    """True if every cell is left-handed in (i, j, k); raise if the orientation is mixed."""

    # signed cell volumes
    volumes = _cell_volumes(xyz)

    # all one way, or folded
    if np.all(volumes > 0.0):
        return False
    if np.all(volumes < 0.0):
        return True
    n_bad = int(np.count_nonzero(volumes <= 0.0))
    raise ValueError(
        f"grid is folded: {n_bad} of {volumes.size} cells have no volume or the opposite "
        "orientation"
    )


def _find_axis(xyz: NDArray[np.float64]) -> bool:
    """True if every k-line of the face i = 0 collapses to a point; refuse other collapses.

    A k-line is collapsed when its length is below COLLAPSE_TOLERANCE times the length
    of the k-line next to it, one step into the grid.
    """

    # every i- and j-side: is it collapsed along k, line by line? (the j-sides without
    # their line at i = 0, which is the imin face's edge: on the axis if imin is)
    collapsed_sides = {}
    for side, face, neighbour in [
        ("imin", xyz[0], xyz[1]),
        ("imax", xyz[-1], xyz[-2]),
        ("jmin", xyz[1:, 0], xyz[1:, 1]),
        ("jmax", xyz[1:, -1], xyz[1:, -2]),
    ]:
        # length of each k-line on the face and next to it
        face_length = np.sum(np.linalg.norm(np.diff(face, axis=1), axis=-1), axis=1)
        neighbour_length = np.sum(np.linalg.norm(np.diff(neighbour, axis=1), axis=-1), axis=1)
        collapsed = face_length <= COLLAPSE_TOLERANCE * neighbour_length
        collapsed_sides[side] = collapsed

    # check: only i = 0 may collapse, and then along its whole length
    imin_collapsed = collapsed_sides.pop("imin")
    for side, collapsed in collapsed_sides.items():
        if np.any(collapsed):
            raise ValueError(
                f"the grid's {side} face collapses onto a line; only an axis at i = 0 "
                "(imin) is supported"
            )
    if np.any(imin_collapsed) and not np.all(imin_collapsed):
        n_collapsed = int(np.count_nonzero(imin_collapsed))
        raise ValueError(
            f"the imin face collapses onto the axis on {n_collapsed} of "
            f"{imin_collapsed.size} k-lines; an axis must collapse on all of them"
        )

    axis = bool(np.all(imin_collapsed))

    return axis


# --------------------------------------------------
# helpers: connectivity
# --------------------------------------------------
def _element_nodes(
    index: NDArray[np.int64],
    i_start: NDArray[np.int64],
    j_start: NDArray[np.int64],
    k_start: NDArray[np.int64],
    offsets: list[tuple[int, int, int]],
) -> NDArray[np.int64]:
    """Structured indices of the nodes of every element, shape (n_elements, len(offsets)).

    Elements run i fastest, then j, then k, like the grid points.
    """

    # first grid point of each element
    i_base, j_base, k_base = np.meshgrid(i_start, j_start, k_start, indexing="ij")
    i_base = i_base.ravel(order="F")
    j_base = j_base.ravel(order="F")
    k_base = k_base.ravel(order="F")

    # one column per element node
    columns = []
    for di, dj, dk in offsets:
        columns.append(index[i_base + di, j_base + dj, k_base + dk])
    nodes = np.column_stack(columns).astype(np.int64)

    return nodes


def _side_faces(
    index: NDArray[np.int64], side: str, order: int, axis: bool
) -> dict[str, NDArray[np.int64]]:
    """Quadrilaterals and triangles on one grid side, as structured indices of their nodes.

    Faces are listed in the side's (a, b) frame, a fastest; their corners turn
    counter-clockwise about the outward normal.
    """

    # the side's frame
    fixed_axis, fixed_end, axis_a, axis_b = _SIDE_FRAMES[side]
    shape = index.shape
    fixed_index = 0 if fixed_end == "first" else shape[fixed_axis] - 1

    # the face i = 0 on the axis has no area: no faces
    n_quad = len(_QUAD_OFFSETS[order])
    n_triangle = len(_AXIS_TRIANGLE_OFFSETS["a"][order])
    empty = {
        "quadrilateral": np.empty((0, n_quad), dtype=np.int64),
        "triangle": np.empty((0, n_triangle), dtype=np.int64),
    }
    if axis and side == "imin":
        return empty

    # first grid point of each face along a and b
    a_start = np.arange(0, shape[axis_a] - 1, order)
    b_start = np.arange(0, shape[axis_b] - 1, order)
    a_base, b_base = np.meshgrid(a_start, b_start, indexing="ij")
    a_base = a_base.ravel(order="F")
    b_base = b_base.ravel(order="F")

    # faces touching the axis: on the j-sides (which run along i and k), those starting
    # at i = 0; the k-sides' faces at i = 0 have a whole edge on the axis, not a point
    touches_axis = np.zeros(a_base.size, dtype=bool)
    i_direction = None
    if axis and fixed_axis == 1 and axis_a == 0:
        i_direction = "a"
        touches_axis = a_base == 0
    if axis and fixed_axis == 1 and axis_b == 0:
        i_direction = "b"
        touches_axis = b_base == 0

    # node indices for a list of (da, db) offsets
    def face_nodes(selected: NDArray[np.bool_], offsets: list[tuple[int, int]]) -> NDArray[np.int64]:
        columns = []
        for da, db in offsets:
            ijk = [None, None, None]
            ijk[fixed_axis] = np.full(int(selected.sum()), fixed_index)
            ijk[axis_a] = a_base[selected] + da
            ijk[axis_b] = b_base[selected] + db
            columns.append(index[ijk[0], ijk[1], ijk[2]])
        nodes = np.column_stack(columns).astype(np.int64)
        return nodes

    # quadrilaterals away from the axis, triangles touching it
    faces = dict(empty)
    faces["quadrilateral"] = face_nodes(~touches_axis, _QUAD_OFFSETS[order])
    if i_direction is not None:
        faces["triangle"] = face_nodes(touches_axis, _AXIS_TRIANGLE_OFFSETS[i_direction][order])

    return faces
