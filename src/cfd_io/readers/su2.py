"""SU2 mesh (``.su2``) reader with structured grid reconstruction.

Reads SU2-native mesh files containing quad (2-D) elements and
reconstructs the original structured (i, j) ordering via a topological
walk, returning a ``StructuredGrid`` of shape ``(ni, nj, 1)``.

The reconstruction reuses the adjacency-walk algorithm from the VTU
reader (``_build_adjacency``, ``_reconstruct_cells``, ``_extract_nodes``),
which is format-independent.

Currently supports 2-D (quad) meshes only.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.readers.vtu import _build_adjacency, _extract_nodes, _reconstruct_cells

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# SU2 file parser
# --------------------------------------------------
def _parse_su2(fpath: Path):
    """Parse SU2 mesh file into elements, points, and boundary markers.

    Args:
        fpath: Path to the ``.su2`` file.

    Returns:
        Tuple of (ndime, conn, points, markers) where:

        - ndime: Number of spatial dimensions.
        - conn: Element connectivity, shape ``(n_cells, nodes_per_cell)``.
        - points: Node coordinates, shape ``(n_points, ndime)``.
        - markers: Dict mapping marker tag name to boundary element
          connectivity arrays, shape ``(n_belem, nodes_per_belem)``.
    """
    with open(fpath) as f:
        lines = f.readlines()

    idx = 0
    ndime = 0
    conn = None
    points = None
    markers: dict[str, np.ndarray] = {}

    while idx < len(lines):
        line = lines[idx].strip()

        # skip blank lines and comments
        if not line or line.startswith("%"):
            idx += 1
            continue

        if line.startswith("NDIME="):
            ndime = int(line.split("=")[1])
            idx += 1

        elif line.startswith("NELEM="):
            nelem = int(line.split("=")[1])
            elem_list = []
            for _ in range(nelem):
                idx += 1
                parts = lines[idx].split()
                # format: etype node0 node1 ... nodeN element_id
                # node IDs are between the type code and the element ID (last value)
                nodes = [int(x) for x in parts[1:-1]]
                elem_list.append(nodes)
            conn = np.array(elem_list, dtype=np.intp)
            idx += 1

        elif line.startswith("NPOIN="):
            # NPOIN can have format "NPOIN= 6" or "NPOIN= 6 0" (some solvers)
            npoin = int(line.split("=")[1].split()[0])
            pts = np.zeros((npoin, ndime))
            for p in range(npoin):
                idx += 1
                parts = lines[idx].split()
                for d in range(ndime):
                    pts[p, d] = float(parts[d])
            points = pts
            idx += 1

        elif line.startswith("NMARK="):
            nmark = int(line.split("=")[1])
            idx += 1
            for _ in range(nmark):
                # MARKER_TAG= name
                tag = lines[idx].strip().split("=")[1].strip()
                idx += 1
                # MARKER_ELEMS= count
                n_belem = int(lines[idx].strip().split("=")[1])
                idx += 1
                belems = []
                for _ in range(n_belem):
                    parts = lines[idx].split()
                    # skip type code, keep node IDs
                    bnodes = [int(x) for x in parts[1:]]
                    belems.append(bnodes)
                    idx += 1
                markers[tag] = np.array(belems, dtype=np.intp)

        else:
            idx += 1

    return ndime, conn, points, markers


# --------------------------------------------------
# structured marker reconstruction
# --------------------------------------------------
def _identify_structured_side(
    belem_nodes: np.ndarray,
    node_grid: np.ndarray,
) -> str | None:
    """Identify which structured side a set of boundary elements belongs to.

    Checks if the boundary nodes lie entirely on one of the four sides:
    imin (i=0), imax (i=ni-1), jmin (j=0), jmax (j=nj-1).

    Args:
        belem_nodes: Boundary element connectivity, shape ``(n_belem, 2)``.
        node_grid: Structured node grid, shape ``(ni, nj)``.

    Returns:
        Side name string, or None if no match.
    """
    all_nodes = set(belem_nodes.ravel())

    sides = {
        "imin": set(node_grid[0, :]),
        "imax": set(node_grid[-1, :]),
        "jmin": set(node_grid[:, 0]),
        "jmax": set(node_grid[:, -1]),
    }

    for side_name, side_nodes in sides.items():
        if all_nodes.issubset(side_nodes):
            return side_name

    return None


# --------------------------------------------------
# public API
# --------------------------------------------------
def read_su2(fpath: str | Path) -> Dataset:
    """Read an SU2 mesh file and reconstruct the structured grid.

    Parses the SU2 text format, then recovers structured (i, j) ordering
    via a topological walk (adjacency graph of shared quad edges).  The
    boundary markers are mapped back to structured side names.

    Only 2-D (quad) meshes are currently supported.

    Args:
        fpath: Path to the ``.su2`` file.

    Returns:
        ``Dataset`` with ``StructuredGrid`` of shape ``(ni, nj, 1)``
        and marker information in ``attrs["markers"]["structured"]``.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        NotImplementedError: If the mesh is 3-D.
        ValueError: If the file is missing required sections or
            contains non-quad elements.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(fpath)

    logger.info("read_su2: %s", fpath)

    # parse the SU2 file
    ndime, conn, points, su2_markers = _parse_su2(fpath)

    # validate
    if ndime != 2:
        raise NotImplementedError(
            f"only 2-D meshes are supported, got NDIME={ndime}"
        )
    if conn is None or points is None:
        raise ValueError("SU2 file is missing NELEM or NPOIN section")

    n_cells = conn.shape[0]
    nodes_per_cell = conn.shape[1]

    if nodes_per_cell != 4:
        raise ValueError(
            f"expected quad elements (4 nodes per cell), got {nodes_per_cell}"
        )

    logger.debug(
        "parsed: %d cells, %d points, %d markers",
        n_cells, points.shape[0], len(su2_markers),
    )

    # --------------------------------------------------
    # reconstruct structured ordering
    # --------------------------------------------------

    # build cell adjacency from shared edges
    neighbors = _build_adjacency(conn, n_cells)

    # pad 2D points to 3D (reconstruction functions use points[:, 0] and [:, 1])
    points_3d = np.column_stack([points, np.zeros(points.shape[0])])

    # recover cell grid (ni, nj) via topological walk
    cell_grid, ni, nj = _reconstruct_cells(conn, n_cells, neighbors, points_3d)

    # extract structured node grid
    node_grid = _extract_nodes(cell_grid, conn, points_3d)

    # flip j so j=0 is at the wall (smaller mean y values)
    # use mean across the full face for robustness (single-point check
    # fails at blunt nose tips where both sides have y≈0)
    y_j0_mean = points[node_grid[:, 0], 1].mean()
    y_jn_mean = points[node_grid[:, -1], 1].mean()
    if y_j0_mean > y_jn_mean:
        node_grid = node_grid[:, ::-1]

    # build StructuredGrid (ni+1, nj+1, 1)
    x = points[node_grid, 0][:, :, np.newaxis]
    y = points[node_grid, 1][:, :, np.newaxis]
    z = np.zeros_like(x)

    # --------------------------------------------------
    # reconstruct structured marker mapping
    # --------------------------------------------------
    structured_markers: dict[str, str] = {}
    for tag, belems in su2_markers.items():
        side = _identify_structured_side(belems, node_grid)
        if side is not None:
            structured_markers[side] = tag

    attrs: dict[str, Any] = {
        "format": "su2",
        "ni": ni + 1,
        "nj": nj + 1,
        "nk": 1,
        "markers": {"structured": structured_markers},
    }

    logger.info(
        "  grid: (%d, %d, 1), %d markers reconstructed",
        ni + 1, nj + 1, len(structured_markers),
    )

    return Dataset(
        grid=StructuredGrid(x=x, y=y, z=z),
        flow={},
        attrs=attrs,
    )
