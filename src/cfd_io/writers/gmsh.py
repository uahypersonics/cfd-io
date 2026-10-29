"""Gmsh mesh (``.msh``) writer, MSH 4.1 ASCII.

Writes 2-D structured grids, shape ``(ni, nj, 1)`` in the x-y plane, as an
unstructured Gmsh mesh of quads:

- **order 1:** 4-node quads, one per grid cell, and 2-node boundary lines.
- **order 2:** 9-node quads built from every other grid point (2 x 2 grid
  cells each), and 3-node boundary lines, so curved walls stay curved.
  Needs odd ``ni`` and ``nj``.

Every grid point becomes a node; node tag = ``1 + i + ni * j``.

**Physical groups**:

- one per boundary name in ``dataset.attrs["markers"]["structured"]``
  (sides that share a name share a group); solvers attach boundary
  conditions by these names
- ``fluid`` for the quads, the conventional name of the volume group

Physical tags are unique across dimensions (boundaries 1..n, ``fluid``
n + 1), so readers that look groups up by tag alone still find the right one.

**Entities**: one surface (the quads) and one curve per grid side, each
carrying its physical group. All nodes sit on the surface entity.

**Orientation**: quads are written counter-clockwise in x-y (positive
Jacobian). A grid whose cells are all clockwise in (i, j)
is written with the node order reversed; a grid with cells of both
orientations (folded) raises.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO

import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.writers._markers import _extract_markers, _validate_markers

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Gmsh element type codes and node layouts
# https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
# --------------------------------------------------

# element type codes per order: (boundary line, quad)
_GMSH_TYPES = {
    1: (1, 3),
    2: (8, 10),
}

# quad nodes as (di, dj) offsets from the element's lower-left grid point, in
# Gmsh order: corners counter-clockwise, then edge midpoints, then the centre
_QUAD_OFFSETS = {
    1: [(0, 0), (1, 0), (1, 1), (0, 1)],
    2: [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)],
}

# boundary line nodes as offsets along the side, in Gmsh order: ends, then middle
_LINE_OFFSETS = {
    1: [0, 1],
    2: [0, 2, 1],
}

# grid sides in the order their curve entities are written (curve tag = position + 1)
_SIDES_2D = ["imin", "imax", "jmin", "jmax"]

# physical name of the volume (here: area) elements
_FLUID_NAME = "fluid"

# coordinates with enough digits to round-trip doubles exactly
_COORD_FMT = "%.17g"


# --------------------------------------------------
# public API
# --------------------------------------------------
def write_gmsh(fpath: str | Path, dataset: Dataset, *, order: int = 1) -> Path:
    """Write a 2-D structured ``Dataset`` to a Gmsh MSH 4.1 ASCII file.

    Args:
        fpath: Output file path.
        dataset: Dataset with a ``StructuredGrid`` of shape ``(ni, nj, 1)`` and
            ``attrs["markers"]["structured"]`` naming imin, imax, jmin, jmax.
        order: Element order: 1 (4-node quads) or 2 (9-node quads, needs odd
            ``ni`` and ``nj``).

    Returns:
        Path to the created file.

    Raises:
        TypeError: If the grid isn't a ``StructuredGrid``, or the markers
            have the wrong type.
        ValueError: If the grid isn't 2-D in the x-y plane, the point counts
            don't suit the order, markers are missing or named ``fluid``, or
            the grid is folded.
    """

    # convert to Path object
    fpath = Path(fpath)

    # validate inputs
    grid = dataset.grid
    if not isinstance(grid, StructuredGrid):
        raise TypeError(f"Gmsh writer requires a StructuredGrid, got {type(grid).__name__}")
    _check_grid_shape(grid.shape, order)

    # boundary names per side
    markers = _extract_markers(dataset.attrs, _SIDES_2D)
    _validate_markers(markers, _SIDES_2D)

    # physical tags: one per distinct boundary name, in side order, then fluid
    boundary_names = list(dict.fromkeys(markers[side] for side in _SIDES_2D))
    if _FLUID_NAME in (name.lower() for name in boundary_names):
        raise ValueError(f"boundary name {_FLUID_NAME!r} is reserved for the quads")
    physical_tags = {name: tag for tag, name in enumerate(boundary_names, start=1)}
    fluid_tag = len(boundary_names) + 1

    # node coordinates in tag order (i fastest)
    ni, nj, _ = grid.shape
    points = np.column_stack(
        [grid.x.ravel(order="F"), grid.y.ravel(order="F"), grid.z.ravel(order="F")]
    )

    # check orientation: reverse the node order of clockwise grids
    clockwise = _is_clockwise(grid.x[:, :, 0], grid.y[:, :, 0])

    # connectivity
    quads = _quad_nodes(ni, nj, order, clockwise)
    side_lines = {side: _side_line_nodes(side, ni, nj, order) for side in _SIDES_2D}

    # write
    with open(fpath, "w") as stream:
        _write_header(stream, boundary_names, physical_tags, fluid_tag)
        _write_entities(stream, grid, markers, physical_tags, fluid_tag)
        _write_nodes(stream, points)
        _write_elements(stream, quads, side_lines, order)

    # debug output for devs
    logger.info(
        "wrote %s  (order %d, %d nodes, %d quads, %d boundary groups)",
        fpath, order, points.shape[0], quads.shape[0], len(boundary_names),
    )

    return fpath


# --------------------------------------------------
# helpers: checks and connectivity
# --------------------------------------------------
def _check_grid_shape(shape: tuple[int, int, int], order: int) -> None:
    """Raise unless the grid is (ni, nj, 1) with point counts that suit *order*."""

    # check the order
    if order not in _QUAD_OFFSETS:
        raise ValueError(f"order must be 1 or 2, got {order}")

    # check: 2-D in the x-y plane
    ni, nj, nk = shape
    if nk != 1 or ni < 2 or nj < 2:
        raise ValueError(
            f"Gmsh writer supports 2-D grids of shape (ni, nj, 1) with ni, nj >= 2; got {shape}"
        )

    # check: second-order elements span two grid cells per direction
    if order == 2 and (ni % 2 == 0 or nj % 2 == 0 or ni < 3 or nj < 3):
        raise ValueError(
            f"order 2 needs odd ni and nj (>= 3), got ni = {ni}, nj = {nj}"
        )


def _is_clockwise(x: np.ndarray, y: np.ndarray) -> bool:
    """True if every grid cell is clockwise in (i, j); raise if the grid is folded."""

    # signed cell areas from the diagonals (positive = counter-clockwise in i, j)
    diag_1_x = x[1:, 1:] - x[:-1, :-1]
    diag_1_y = y[1:, 1:] - y[:-1, :-1]
    diag_2_x = x[:-1, 1:] - x[1:, :-1]
    diag_2_y = y[:-1, 1:] - y[1:, :-1]
    area = 0.5 * (diag_1_x * diag_2_y - diag_1_y * diag_2_x)

    # all one way, or folded
    if np.all(area > 0.0):
        return False
    if np.all(area < 0.0):
        return True
    n_bad = int(np.count_nonzero(area <= 0.0))
    raise ValueError(
        f"grid is folded: {n_bad} of {area.size} cells have zero or opposite orientation"
    )


def _quad_nodes(ni: int, nj: int, order: int, clockwise: bool) -> np.ndarray:
    """Node tags of every quad, shape (n_quads, 4 or 9), counter-clockwise in x-y."""

    # lower-left grid point of each element (elements span `order` grid cells)
    i_start = np.arange(0, ni - 1, order)
    j_start = np.arange(0, nj - 1, order)
    i_base, j_base = np.meshgrid(i_start, j_start, indexing="ij")
    i_base = i_base.ravel(order="F")
    j_base = j_base.ravel(order="F")

    # a clockwise grid becomes counter-clockwise by swapping the local i and j offsets
    offsets = _QUAD_OFFSETS[order]
    if clockwise:
        offsets = [(dj, di) for di, dj in offsets]

    # one column per element node
    columns = []
    for di, dj in offsets:
        node_tag = 1 + (i_base + di) + ni * (j_base + dj)
        columns.append(node_tag)
    quads = np.column_stack(columns)

    return quads


def _side_line_nodes(side: str, ni: int, nj: int, order: int) -> np.ndarray:
    """Node tags of the boundary lines along one grid side, shape (n_lines, 2 or 3)."""

    # grid points along the side: (i, j) of the first point and the step between points
    if side == "imin":
        i_first, j_first, step_i, step_j, n_side = 0, 0, 0, 1, nj
    elif side == "imax":
        i_first, j_first, step_i, step_j, n_side = ni - 1, 0, 0, 1, nj
    elif side == "jmin":
        i_first, j_first, step_i, step_j, n_side = 0, 0, 1, 0, ni
    else:
        i_first, j_first, step_i, step_j, n_side = 0, nj - 1, 1, 0, ni

    # first point of each line along the side
    line_start = np.arange(0, n_side - 1, order)

    # one column per line node
    columns = []
    for offset in _LINE_OFFSETS[order]:
        position = line_start + offset
        node_tag = 1 + (i_first + step_i * position) + ni * (j_first + step_j * position)
        columns.append(node_tag)
    lines = np.column_stack(columns)

    return lines


# --------------------------------------------------
# helpers: file sections
# --------------------------------------------------
def _write_header(
    stream: TextIO,
    boundary_names: list[str],
    physical_tags: dict[str, int],
    fluid_tag: int,
) -> None:
    """Write $MeshFormat and $PhysicalNames."""

    # format: version 4.1, ASCII (0), size_t of 8 bytes
    stream.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")

    # physical names: boundaries (dimension 1), then the fluid (dimension 2)
    stream.write("$PhysicalNames\n")
    stream.write(f"{len(boundary_names) + 1}\n")
    for name in boundary_names:
        stream.write(f'1 {physical_tags[name]} "{name}"\n')
    stream.write(f'2 {fluid_tag} "{_FLUID_NAME}"\n')
    stream.write("$EndPhysicalNames\n")


def _write_entities(
    stream: TextIO,
    grid: StructuredGrid,
    markers: dict[str, str],
    physical_tags: dict[str, int],
    fluid_tag: int,
) -> None:
    """Write $Entities: one curve per grid side, one surface; no points, no volumes."""

    # grid points on each side
    side_slices = {
        "imin": (0, slice(None)),
        "imax": (-1, slice(None)),
        "jmin": (slice(None), 0),
        "jmax": (slice(None), -1),
    }

    # counts: points, curves, surfaces, volumes
    stream.write("$Entities\n")
    stream.write(f"0 {len(_SIDES_2D)} 1 0\n")

    # curves: tag, bounding box, one physical tag, no bounding points
    for curve_tag, side in enumerate(_SIDES_2D, start=1):
        i_index, j_index = side_slices[side]
        box = _bounding_box(
            grid.x[i_index, j_index, 0], grid.y[i_index, j_index, 0], grid.z[i_index, j_index, 0]
        )
        physical_tag = physical_tags[markers[side]]
        stream.write(f"{curve_tag} {box} 1 {physical_tag} 0\n")

    # surface: tag, bounding box, the fluid physical tag, bounded by the four curves
    box = _bounding_box(grid.x, grid.y, grid.z)
    curve_tags = " ".join(str(tag) for tag in range(1, len(_SIDES_2D) + 1))
    stream.write(f"1 {box} 1 {fluid_tag} {len(_SIDES_2D)} {curve_tags}\n")
    stream.write("$EndEntities\n")


def _write_nodes(stream: TextIO, points: np.ndarray) -> None:
    """Write $Nodes: one block on the surface entity, tags 1..n."""

    # header: blocks, nodes, min and max tag; block: dimension, entity, not parametric, count
    n_nodes = points.shape[0]
    stream.write("$Nodes\n")
    stream.write(f"1 {n_nodes} 1 {n_nodes}\n")
    stream.write(f"2 1 0 {n_nodes}\n")

    # tags, then coordinates
    np.savetxt(stream, np.arange(1, n_nodes + 1), fmt="%d")
    np.savetxt(stream, points, fmt=_COORD_FMT)
    stream.write("$EndNodes\n")


def _write_elements(
    stream: TextIO,
    quads: np.ndarray,
    side_lines: dict[str, np.ndarray],
    order: int,
) -> None:
    """Write $Elements: one block of lines per side curve, then the quads on the surface."""

    # element types and total count
    line_type, quad_type = _GMSH_TYPES[order]
    n_lines = sum(lines.shape[0] for lines in side_lines.values())
    n_elements = n_lines + quads.shape[0]

    # header: blocks, elements, min and max tag
    stream.write("$Elements\n")
    stream.write(f"{len(_SIDES_2D) + 1} {n_elements} 1 {n_elements}\n")

    # boundary lines, one block per side curve
    next_tag = 1
    for curve_tag, side in enumerate(_SIDES_2D, start=1):
        lines = side_lines[side]
        stream.write(f"1 {curve_tag} {line_type} {lines.shape[0]}\n")
        next_tag = _write_element_rows(stream, lines, next_tag)

    # quads on the surface
    stream.write(f"2 1 {quad_type} {quads.shape[0]}\n")
    _write_element_rows(stream, quads, next_tag)
    stream.write("$EndElements\n")


def _write_element_rows(stream: TextIO, nodes: np.ndarray, first_tag: int) -> int:
    """Write element tag + node tags per row; return the next free element tag."""

    # prepend element tags
    n_rows = nodes.shape[0]
    element_tags = np.arange(first_tag, first_tag + n_rows)
    rows = np.column_stack([element_tags, nodes])

    # write
    np.savetxt(stream, rows, fmt="%d")

    return first_tag + n_rows


def _bounding_box(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> str:
    """Return 'min_x min_y min_z max_x max_y max_z' for an entity line."""

    # min and max per coordinate
    lower = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
    upper = [float(np.max(x)), float(np.max(y)), float(np.max(z))]
    box = " ".join(_COORD_FMT % value for value in lower + upper)

    return box
