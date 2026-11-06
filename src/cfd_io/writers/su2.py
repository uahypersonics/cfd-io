"""SU2 mesh (``.su2``) writer.

Writes SU2-native mesh files from structured grids.  The writer converts
the structured (i, j, k) grid to an unstructured node/element
representation and serializes it to the SU2 plain-text format.

Supports 2-D (quad) and 3-D (hex) structured grids.  Boundary markers
are specified via ``dataset.attrs["markers"]["structured"]``, which maps
logical side names (``imin``, ``imax``, ``jmin``, ``jmax``,
``kmin``, ``kmax``) to SU2 marker tag strings.

**Axis** (3-D) — a grid revolved about an axis, with its face ``i = 0`` on
the axis (each k-line of it collapsed to a point), gets one node per
collapsed line, prisms for the cells touching the axis, and triangles for
the boundary faces touching it.  The face ``i = 0`` has no area and no
marker in the file: the axis is an edge inside the mesh (see
``cfd_io.writers._volume_mesh``).  A left-handed 3-D grid is written with
k reversed (``kmin`` and ``kmax`` swap planes, keeping their names on the
same faces); a folded one is refused.

**Marker policy** — strict by default:

- 2-D requires all 4 perimeter sides for the active axes.
- 3-D requires all 6 sides.
- Missing keys raise ``ValueError``.

**Dimensionality rule** — determined by ``StructuredGrid.ndim``:

- ``(ni, nj, 1)`` → 2-D (``NDIME=2``).
- ``(ni, 1, nk)`` → 2-D (``NDIME=2``).
- ``(ni, nj, nk)`` all > 1 → 3-D (``NDIME=3``).
- 1-D grids are not supported.

**Indexing**: all node and element IDs are 0-based (SU2 native).

**Node ordering**: SU2-native (per SU2 documentation).

**Coordinate format**: ``%.15e`` for deterministic double-precision output.

**Scope**: structured grids only.  ``UnstructuredGrid`` raises
``NotImplementedError``.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid, UnstructuredGrid
from cfd_io.writers._markers import (
    _SIDES_FOR_AXIS,
    _extract_markers,
    _required_sides,
    _validate_markers,
)
from cfd_io.writers._volume_mesh import SIDES_3D, build_volume_mesh

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# SU2 element type codes (from SU2 documentation)
# https://su2code.github.io/docs_v7/Mesh-File/
# --------------------------------------------------
_SU2_LINE = 3
_SU2_TRIANGLE = 5
_SU2_QUAD = 9
_SU2_HEX = 12
_SU2_PRISM = 13

# SU2 prisms follow VTK's wedge, whose first triangle faces away from the second;
# Gmsh's (the volume mesh's order) faces toward it: swap the second and third nodes
# of both triangles
_VTK_PRISM_ORDER = [0, 2, 1, 3, 5, 4]

# --------------------------------------------------
# Coordinate format (decided once, used everywhere)
# --------------------------------------------------
_COORD_FMT = "%.15e"


# --------------------------------------------------
# helper function to count number of axes with size > 1
# --------------------------------------------------
def _get_active_axes(shape: tuple[int, ...]) -> list[int]:
    """Return indices of axes with size > 1."""
    return [a for a in range(len(shape)) if shape[a] > 1]


# --------------------------------------------------
# helper function to merge boundary elements by marker tag
# --------------------------------------------------
def _merge_boundaries(
    boundary_raw: dict[str, dict[int, np.ndarray]],
    markers: dict[str, str],
    canonical_order: list[str],
) -> dict[str, dict[int, np.ndarray]]:
    """Group boundary element blocks by marker tag name.

    Iterates *canonical_order*, maps each side to its tag via *markers*,
    and concatenates the blocks of one element type that share the same
    tag.  The returned dict preserves first-seen order, of tags and of
    element types within a tag.
    """
    groups: dict[str, dict[int, list[np.ndarray]]] = {}

    for side in canonical_order:
        if side not in boundary_raw:
            continue
        tag = markers[side]
        if tag not in groups:
            groups[tag] = {}
        for etype, nodes in boundary_raw[side].items():
            if etype not in groups[tag]:
                groups[tag][etype] = []
            groups[tag][etype].append(nodes)

    result: dict[str, dict[int, np.ndarray]] = {}
    for name, blocks in groups.items():
        result[name] = {}
        for etype, arrs in blocks.items():
            result[name][etype] = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=0)
    return result


# --------------------------------------------------
# structured to unstructured conversion function (receive structured grid + markers, return unstructured points/elements/boundary_elements)
# --------------------------------------------------
def structured_to_unstructured(
    grid,
    markers: dict[str, str],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Convert a structured grid to an unstructured representation.

    Args:
        grid: ``StructuredGrid`` with shape ``(ni, nj, nk)``.
        markers: Maps logical side names (``imin``, ``jmax``, …) to
            SU2 marker tag strings.

    Returns:
        ``(points, elements, boundary_elements)`` where

        - *points*: ``(npoints, ndime)`` — coordinate array.
          Columns are ``(x, y)`` for 2-D, ``(x, y, z)`` for 3-D.
        - *elements*: ``dict[int, ndarray]`` — element connectivity
          (area cells in 2-D, volume cells in 3-D) keyed by SU2 element
          type: quads (9) in 2-D; hexes (12) in 3-D, plus prisms (13) at
          an axis.
        - *boundary_elements*: ``dict[str, dict[int, ndarray]]`` —
          boundary element blocks keyed by SU2 marker tag, then by SU2
          element type: line segments (3) in 2-D; quad faces (9) in 3-D,
          plus triangles (5) touching an axis.

    Raises:
        TypeError: If *grid* is not a ``StructuredGrid``.
        ValueError: If dimensionality is unsupported, grid is too small
            or folded (3-D), or markers are incomplete.
    """

    if not isinstance(grid, StructuredGrid):
        raise TypeError(
            f"expected StructuredGrid, got {type(grid).__name__}"
        )

    shape = grid.shape
    ndim = grid.ndim

    if ndim < 2:
        raise ValueError(
            f"SU2 writer requires at least a 2-D grid; "
            f"grid.ndim={ndim} (shape={shape})"
        )

    active_axes = _get_active_axes(shape)

    # Validate minimum size on active axes
    for ax in active_axes:
        if shape[ax] < 2:
            raise ValueError(
                f"active axis {ax} has size {shape[ax]} < 2; "
                f"cannot form cells"
            )

    required = _required_sides(active_axes)
    _validate_markers(markers, required)

    if ndim == 2:
        return _structured_to_unstructured_2d(grid, markers, active_axes)
    else:
        return _structured_to_unstructured_3d(grid, markers)


# --------------------------------------------------
# convert structured to unstructured in 2d
# --------------------------------------------------
def _structured_to_unstructured_2d(
    grid,
    markers: dict[str, str],
    active_axes: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """2-D: quads + line boundary segments."""
    shape = grid.shape
    coords_3d = [grid.x, grid.y, grid.z]

    # Identify degenerate and active axes
    degen_ax = [a for a in range(3) if a not in active_axes][0]
    ax0, ax1 = active_axes
    n0, n1 = shape[ax0], shape[ax1]

    # ── Points ───────────────────────────────────────────────────────
    # Drop the coordinate for the degenerate axis
    coord_indices = [a for a in range(3) if a != degen_ax]
    slc = [slice(None)] * 3
    slc[degen_ax] = 0

    coord_arrays = []
    for ci in coord_indices:
        arr = coords_3d[ci][tuple(slc)]       # shape (n0, n1)
        coord_arrays.append(arr.ravel(order="F"))

    points = np.column_stack(coord_arrays)

    # ── Volume elements (quads, SU2-native CCW) ─────────────────────
    # Flat node ID: a + n0 * b  (F-order on the 2-D sub-array)
    a_idx = np.arange(n0 - 1)
    b_idx = np.arange(n1 - 1)
    aa, bb = np.meshgrid(a_idx, b_idx, indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()

    elements = np.column_stack([
        aa + n0 * bb,                  # (a,   b  )
        (aa + 1) + n0 * bb,            # (a+1, b  )
        (aa + 1) + n0 * (bb + 1),      # (a+1, b+1)
        aa + n0 * (bb + 1),            # (a,   b+1)
    ])

    # ── Boundary segments (lines) ────────────────────────────────────
    side_lo_0, side_hi_0 = _SIDES_FOR_AXIS[ax0]
    side_lo_1, side_hi_1 = _SIDES_FOR_AXIS[ax1]

    b_range = np.arange(n1 - 1)
    a_range = np.arange(n0 - 1)

    boundary_raw: dict[str, np.ndarray] = {}

    # ax0 min (a=0), traverse b
    boundary_raw[side_lo_0] = np.column_stack([
        np.full_like(b_range, 0) + n0 * b_range,
        np.full_like(b_range, 0) + n0 * (b_range + 1),
    ])
    # ax0 max (a=n0-1), traverse b
    boundary_raw[side_hi_0] = np.column_stack([
        np.full_like(b_range, n0 - 1) + n0 * b_range,
        np.full_like(b_range, n0 - 1) + n0 * (b_range + 1),
    ])
    # ax1 min (b=0), traverse a
    boundary_raw[side_lo_1] = np.column_stack([
        a_range + n0 * 0,
        (a_range + 1) + n0 * 0,
    ])
    # ax1 max (b=n1-1), traverse a
    boundary_raw[side_hi_1] = np.column_stack([
        a_range + n0 * (n1 - 1),
        (a_range + 1) + n0 * (n1 - 1),
    ])

    # ── Merge by marker name ─────────────────────────────────────────
    canonical = _required_sides(active_axes)
    boundary_blocks = {}
    for side, lines in boundary_raw.items():
        boundary_blocks[side] = {_SU2_LINE: lines}
    boundary_elements = _merge_boundaries(boundary_blocks, markers, canonical)

    # one element type in 2-D
    element_blocks = {_SU2_QUAD: elements}

    return points, element_blocks, boundary_elements


# --------------------------------------------------
# convert structured to unstructured in 3d
# --------------------------------------------------
def _structured_to_unstructured_3d(
    grid,
    markers: dict[str, str],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """3-D: hexes (prisms at an axis) + quad (and triangle) face boundaries."""

    # build the volume mesh (merges an axis, orients the cells right-handed)
    mesh = build_volume_mesh(grid, order=1)

    # volume elements: hexes, then the prisms at an axis in VTK's order
    elements = {_SU2_HEX: mesh.hexahedra}
    if mesh.prisms.shape[0] > 0:
        elements[_SU2_PRISM] = mesh.prisms[:, _VTK_PRISM_ORDER]

    # side names: with k reversed, the kmin faces lie on the grid's last k-plane
    side_names = dict(markers)
    if mesh.k_reversed:
        side_names["kmin"] = markers["kmax"]
        side_names["kmax"] = markers["kmin"]

    # boundary faces per side; the side on an axis has none and is left out
    boundary_raw: dict[str, dict[int, np.ndarray]] = {}
    for side in SIDES_3D:
        faces = mesh.faces[side]
        blocks = {}
        if faces["quadrilateral"].shape[0] > 0:
            blocks[_SU2_QUAD] = faces["quadrilateral"]
        if faces["triangle"].shape[0] > 0:
            blocks[_SU2_TRIANGLE] = faces["triangle"]
        if blocks:
            boundary_raw[side] = blocks

    # debug output for devs
    if mesh.axis:
        logger.info(
            "imin lies on an axis: %d prisms around it, no boundary faces for %r",
            mesh.prisms.shape[0], markers["imin"],
        )

    # merge by marker name
    boundary_elements = _merge_boundaries(boundary_raw, side_names, SIDES_3D)

    return mesh.points, elements, boundary_elements


# --------------------------------------------------
# su2 writer
# --------------------------------------------------
def _write_su2_text(
    fpath: Path,
    ndime: int,
    points: np.ndarray,
    elements: dict[int, np.ndarray],
    boundary_elements: dict[str, dict[int, np.ndarray]],
) -> None:
    """Serialize the unstructured representation to SU2 plain text.

    Args:
        fpath: Output file path.
        ndime: Number of spatial dimensions (2 or 3).
        points: Node coordinates, shape ``(npoints, ndime)``.
        elements: Element connectivity keyed by SU2 element type (9 = quad
            in 2-D; 12 = hex, 13 = prism in 3-D), each of shape
            ``(nelem, nodes_per_elem)``.  Global 0-based node IDs in
            SU2-native order.
        boundary_elements: Boundary element blocks keyed by marker tag
            name, then by SU2 element type (3 = line in 2-D; 9 = quad face,
            5 = triangle in 3-D).

    Example:
        Output for a 3×2 structured grid (2 quads, 6 nodes)::

            NDIME= 2
            NELEM= 2
            9 0 1 4 3 0
            9 1 2 5 4 1
            NPOIN= 6
            0.000000000000000e+00 0.000000000000000e+00 0
            1.000000000000000e+00 0.000000000000000e+00 1
            2.000000000000000e+00 0.000000000000000e+00 2
            0.000000000000000e+00 1.000000000000000e+00 3
            1.000000000000000e+00 1.000000000000000e+00 4
            2.000000000000000e+00 1.000000000000000e+00 5
            NMARK= 4
            MARKER_TAG= inlet
            MARKER_ELEMS= 1
            3 0 3
            MARKER_TAG= outlet
            MARKER_ELEMS= 1
            3 2 5
            MARKER_TAG= wall
            MARKER_ELEMS= 2
            3 0 1
            3 1 2
            MARKER_TAG= farfield
            MARKER_ELEMS= 2
            3 3 4
            3 4 5

        Node layout for the example above::

            3 --- 4 --- 5
            | E0  | E1  |
            0 --- 1 --- 2
    """

    # get number of elements and points from shapes of the input arrays
    nelem = sum(nodes.shape[0] for nodes in elements.values())
    npoints = points.shape[0]

    # open file and write su2 format sections
    with open(fpath, "w") as f:

        # grid dimension
        f.write(f"NDIME= {ndime}\n")

        # elements (area cells in 2-D, volume cells in 3-D), block by block
        # this writes: element type, node IDs, element ID (0-based)
        f.write(f"NELEM= {nelem}\n")
        first_id = 0
        for etype, nodes in elements.items():
            n_block = nodes.shape[0]
            element_ids = np.arange(first_id, first_id + n_block)
            rows = np.column_stack([np.full(n_block, etype), nodes, element_ids])
            np.savetxt(f, rows, fmt="%d")
            first_id += n_block

        # points
        # write node coordinates and node ID (0-based)
        f.write(f"NPOIN= {npoints}\n")
        point_rows = np.column_stack([points, np.arange(npoints)])
        point_fmt = [_COORD_FMT] * points.shape[1] + ["%d"]
        np.savetxt(f, point_rows, fmt=point_fmt)

        # markers: element type and node IDs per boundary element
        f.write(f"NMARK= {len(boundary_elements)}\n")
        for tag, blocks in boundary_elements.items():
            n_marker = sum(nodes.shape[0] for nodes in blocks.values())
            f.write(f"MARKER_TAG= {tag}\n")
            f.write(f"MARKER_ELEMS= {n_marker}\n")
            for etype, nodes in blocks.items():
                rows = np.column_stack([np.full(nodes.shape[0], etype), nodes])
                np.savetxt(f, rows, fmt="%d")


# --------------------------------------------------
# public api: SU2 mesh writer
# --------------------------------------------------
def write_su2(fpath: str | Path, dataset: Dataset) -> Path:
    """Write a ``Dataset`` to an SU2 mesh (``.su2``) file.

    Args:
        fpath: Output file path.
        dataset: Dataset to write.  Must have a ``StructuredGrid`` and
            ``attrs["markers"]["structured"]`` with all required side keys.

    Returns:
        Path to the created file.

    Raises:
        TypeError: If the grid is not a ``StructuredGrid``, or the markers
            schema has the wrong type.
        ValueError: If markers are incomplete or the grid is unsupported.
        NotImplementedError: If the grid is an ``UnstructuredGrid``.
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    if isinstance(dataset.grid, UnstructuredGrid):
        raise NotImplementedError(
            "SU2 writer for UnstructuredGrid is not yet implemented"
        )
    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError(
            f"expected StructuredGrid or UnstructuredGrid, "
            f"got {type(dataset.grid).__name__}"
        )

    # Extract and validate markers
    ndim = dataset.grid.ndim
    active_axes = _get_active_axes(dataset.grid.x.shape)
    required = _required_sides(active_axes)
    markers = _extract_markers(dataset.attrs, required)

    # Convert to unstructured representation
    points, elements, boundary_elements = structured_to_unstructured(
        dataset.grid, markers
    )

    # write to su2 text file (the element types travel with the blocks)
    _write_su2_text(fpath, ndim, points, elements, boundary_elements)

    # debug output for devs
    n_elements = sum(nodes.shape[0] for nodes in elements.values())
    logger.info(
        "wrote %s  (NDIME=%d, NELEM=%d, NPOIN=%d, NMARK=%d)",
        fpath, ndim, n_elements, points.shape[0],
        len(boundary_elements),
    )

    return fpath
