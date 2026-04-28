"""SU2 mesh (``.su2``) writer.

Writes SU2-native mesh files from structured grids.  The writer converts
the structured (i, j, k) grid to an unstructured node/element
representation and serializes it to the SU2 plain-text format.

Supports 2-D (quad) and 3-D (hex) structured grids.  Boundary markers
are specified via ``dataset.attrs["markers"]["structured"]``, which maps
logical side names (``imin``, ``imax``, ``jmin``, ``jmax``,
``kmin``, ``kmax``) to SU2 marker tag strings.

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

from cfd_io.dataset import StructuredGrid, UnstructuredGrid

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# SU2 element type codes (from SU2 documentation)
# https://su2code.github.io/docs_v7/Mesh-File/
# --------------------------------------------------
_SU2_LINE = 3
_SU2_QUAD = 9
_SU2_HEX = 12

# --------------------------------------------------
# Canonical side ordering
# --------------------------------------------------
_SIDES_FOR_AXIS = {
    0: ("imin", "imax"),
    1: ("jmin", "jmax"),
    2: ("kmin", "kmax"),
}

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
# helper function to validate markers and merge boundary elements by marker tag
# --------------------------------------------------
def _required_sides(active_axes: list[int]) -> list[str]:
    """Required marker side names in canonical order for *active_axes*."""
    sides: list[str] = []
    for ax in sorted(active_axes):
        lo, hi = _SIDES_FOR_AXIS[ax]
        sides.append(lo)
        sides.append(hi)
    return sides


# --------------------------------------------------
# helper function to validate that all required marker keys are present in the markers dict
# --------------------------------------------------
def _validate_markers(markers: dict, required: list[str]) -> None:
    """Check that *markers* contains every key in *required*.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [s for s in required if s not in markers]
    if missing:
        raise ValueError(
            f"missing required marker keys: {missing}; "
            f"required for this dimensionality: {required}"
        )


# --------------------------------------------------
# helper function to extract and validate markers from dataset.attrs
# --------------------------------------------------
def _extract_markers(attrs: dict, required: list[str]) -> dict[str, str]:
    """Extract ``attrs["markers"]["structured"]``.

    Raises:
        TypeError: ``markers`` is missing or malformed.
    """
    if "markers" not in attrs:
        example = {s: s for s in required}
        raise TypeError(
            "dataset.attrs is missing 'markers'; SU2 writer requires "
            "boundary tag names. Provide them as e.g.\n"
            f"    dataset.attrs['markers'] = {{'structured': {example}}}\n"
            "where each value is the SU2 MARKER_TAG string for that side."
        )
    markers_top = attrs["markers"]
    if not isinstance(markers_top, dict):
        raise TypeError(
            f"attrs['markers'] must be a dict, "
            f"got {type(markers_top).__name__}"
        )
    if "structured" not in markers_top:
        raise TypeError(
            "attrs['markers'] must contain a 'structured' key "
            "for structured grids"
        )
    structured = markers_top["structured"]
    if not isinstance(structured, dict):
        raise TypeError(
            f"attrs['markers']['structured'] must be a dict, "
            f"got {type(structured).__name__}"
        )
    return structured


# --------------------------------------------------
# helper function to merge boundary elements by marker tag
# --------------------------------------------------
def _merge_boundaries(
    boundary_raw: dict[str, np.ndarray],
    markers: dict[str, str],
    canonical_order: list[str],
) -> dict[str, np.ndarray]:
    """Group boundary arrays by marker tag name.

    Iterates *canonical_order*, maps each side to its tag via *markers*,
    and concatenates arrays that share the same tag.  The returned dict
    preserves first-seen order.
    """
    groups: dict[str, list[np.ndarray]] = {}
    ordered_names: list[str] = []

    for side in canonical_order:
        if side not in boundary_raw:
            continue
        tag = markers[side]
        if tag not in groups:
            groups[tag] = []
            ordered_names.append(tag)
        groups[tag].append(boundary_raw[side])

    result: dict[str, np.ndarray] = {}
    for name in ordered_names:
        arrs = groups[name]
        result[name] = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=0)
    return result


# --------------------------------------------------
# structured to unstructured conversion function (receive structured grid + markers, return unstructured points/elements/boundary_elements)
# --------------------------------------------------
def structured_to_unstructured(
    grid,
    markers: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert a structured grid to an unstructured representation.

    Args:
        grid: ``StructuredGrid`` with shape ``(ni, nj, nk)``.
        markers: Maps logical side names (``imin``, ``jmax``, …) to
            SU2 marker tag strings.

    Returns:
        ``(points, elements, boundary_elements)`` where

        - *points*: ``(npoints, ndime)`` — coordinate array.
          Columns are ``(x, y)`` for 2-D, ``(x, y, z)`` for 3-D.
        - *elements*: ``(nelem, nodes_per_elem)`` — element connectivity
          (area cells in 2-D, volume cells in 3-D).
          4 columns (quad) for 2-D, 8 columns (hex) for 3-D.
        - *boundary_elements*: ``dict[str, ndarray]`` — boundary element
          arrays keyed by SU2 marker tag.  ``(n_belem, 2)`` for 2-D
          line segments, ``(n_belem, 4)`` for 3-D quad faces.

    Raises:
        TypeError: If *grid* is not a ``StructuredGrid``.
        ValueError: If dimensionality is unsupported, grid is too small,
            or markers are incomplete.
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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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
    boundary_elements = _merge_boundaries(boundary_raw, markers, canonical)

    return points, elements, boundary_elements


# --------------------------------------------------
# convert structured to unstructured in 3d
# --------------------------------------------------
def _structured_to_unstructured_3d(
    grid,
    markers: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """3-D: hexes + quad face boundaries."""
    ni, nj, nk = grid.shape

    def nid(i, j, k):
        return i + ni * j + ni * nj * k

    # ── Points ───────────────────────────────────────────────────────
    points = np.column_stack([
        grid.x.ravel(order="F"),
        grid.y.ravel(order="F"),
        grid.z.ravel(order="F"),
    ])

    # ── Volume elements (hexes, SU2-native ordering) ─────────────────
    ii, jj, kk = np.meshgrid(
        np.arange(ni - 1), np.arange(nj - 1), np.arange(nk - 1),
        indexing="ij",
    )
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    elements = np.column_stack([
        nid(ii,     jj,     kk),        # 0
        nid(ii + 1, jj,     kk),        # 1
        nid(ii + 1, jj + 1, kk),        # 2
        nid(ii,     jj + 1, kk),        # 3
        nid(ii,     jj,     kk + 1),    # 4
        nid(ii + 1, jj,     kk + 1),    # 5
        nid(ii + 1, jj + 1, kk + 1),    # 6
        nid(ii,     jj + 1, kk + 1),    # 7
    ])

    # ── Boundary faces (quads) ───────────────────────────────────────
    boundary_raw: dict[str, np.ndarray] = {}

    # imin (i=0)
    jb, kb = np.meshgrid(np.arange(nj - 1), np.arange(nk - 1), indexing="ij")
    jb, kb = jb.ravel(), kb.ravel()
    boundary_raw["imin"] = np.column_stack([
        nid(0, jb, kb), nid(0, jb, kb + 1),
        nid(0, jb + 1, kb + 1), nid(0, jb + 1, kb),
    ])

    # imax (i=ni-1)
    boundary_raw["imax"] = np.column_stack([
        nid(ni - 1, jb, kb), nid(ni - 1, jb + 1, kb),
        nid(ni - 1, jb + 1, kb + 1), nid(ni - 1, jb, kb + 1),
    ])

    # jmin (j=0)
    ib, kb2 = np.meshgrid(np.arange(ni - 1), np.arange(nk - 1), indexing="ij")
    ib, kb2 = ib.ravel(), kb2.ravel()
    boundary_raw["jmin"] = np.column_stack([
        nid(ib, 0, kb2), nid(ib + 1, 0, kb2),
        nid(ib + 1, 0, kb2 + 1), nid(ib, 0, kb2 + 1),
    ])

    # jmax (j=nj-1)
    boundary_raw["jmax"] = np.column_stack([
        nid(ib, nj - 1, kb2), nid(ib, nj - 1, kb2 + 1),
        nid(ib + 1, nj - 1, kb2 + 1), nid(ib + 1, nj - 1, kb2),
    ])

    # kmin (k=0)
    ib2, jb2 = np.meshgrid(np.arange(ni - 1), np.arange(nj - 1), indexing="ij")
    ib2, jb2 = ib2.ravel(), jb2.ravel()
    boundary_raw["kmin"] = np.column_stack([
        nid(ib2, jb2, 0), nid(ib2, jb2 + 1, 0),
        nid(ib2 + 1, jb2 + 1, 0), nid(ib2 + 1, jb2, 0),
    ])

    # kmax (k=nk-1)
    boundary_raw["kmax"] = np.column_stack([
        nid(ib2, jb2, nk - 1), nid(ib2 + 1, jb2, nk - 1),
        nid(ib2 + 1, jb2 + 1, nk - 1), nid(ib2, jb2 + 1, nk - 1),
    ])

    # ── Merge by marker name ─────────────────────────────────────────
    canonical = list(_SIDES_FOR_AXIS[0]) + list(_SIDES_FOR_AXIS[1]) + list(_SIDES_FOR_AXIS[2])
    boundary_elements = _merge_boundaries(boundary_raw, markers, canonical)

    return points, elements, boundary_elements


# --------------------------------------------------
# su2 writer
# --------------------------------------------------
def _write_su2_text(
    fpath: Path,
    ndime: int,
    points: np.ndarray,
    elements: np.ndarray,
    elem_etype: int,
    boundary_elements: dict[str, np.ndarray],
    bnd_etype: int,
) -> None:
    """Serialize the unstructured representation to SU2 plain text.

    Args:
        fpath: Output file path.
        ndime: Number of spatial dimensions (2 or 3).
        points: Node coordinates, shape ``(npoints, ndime)``.
        elements: Element connectivity, shape ``(nelem, nodes_per_elem)``.
            Global 0-based node IDs in SU2-native (CCW) order.
        elem_etype: SU2 element type code for domain elements
            (9 = quad in 2-D, 12 = hex in 3-D).
        boundary_elements: Boundary element arrays keyed by marker tag
            name.  Each value has shape ``(n_belem, nodes_per_belem)``.
        bnd_etype: SU2 element type code for boundary elements
            (3 = line in 2-D, 9 = quad face in 3-D).

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
    nelem = elements.shape[0]
    npoints = points.shape[0]

    # open file and write su2 format sections
    with open(fpath, "w") as f:

        # grid dimension
        f.write(f"NDIME= {ndime}\n")

        # elements (area cells in 2-D, volume cells in 3-D)
        # this writes: element type, node IDs, element ID (0-based)
        f.write(f"NELEM= {nelem}\n")
        for eid in range(nelem):
            nodes_str = " ".join(str(int(n)) for n in elements[eid])
            f.write(f"{elem_etype} {nodes_str} {eid}\n")

        # points
        # write node coordinates and node ID (0-based)
        f.write(f"NPOIN= {npoints}\n")
        for pid in range(npoints):
            coords_str = " ".join(_COORD_FMT % c for c in points[pid])
            f.write(f"{coords_str} {pid}\n")

        # markers
        f.write(f"NMARK= {len(boundary_elements)}\n")
        for tag, belems in boundary_elements.items():
            f.write(f"MARKER_TAG= {tag}\n")
            f.write(f"MARKER_ELEMS= {belems.shape[0]}\n")
            for row in belems:
                nodes_str = " ".join(str(int(n)) for n in row)
                f.write(f"{bnd_etype} {nodes_str}\n")


# --------------------------------------------------
# public api: SU2 mesh writer
# --------------------------------------------------
def write_su2(fpath: str | Path, dataset) -> Path:
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

    # Determine SU2 element types
    if ndim == 2:
        elem_etype = _SU2_QUAD
        bnd_etype = _SU2_LINE
    else:
        elem_etype = _SU2_HEX
        bnd_etype = _SU2_QUAD

    # write to su2 text file
    _write_su2_text(
        fpath, ndim, points, elements, elem_etype,
        boundary_elements, bnd_etype,
    )

    logger.info(
        "wrote %s  (NDIME=%d, NELEM=%d, NPOIN=%d, NMARK=%d)",
        fpath, ndim, elements.shape[0], points.shape[0],
        len(boundary_elements),
    )

    return fpath
