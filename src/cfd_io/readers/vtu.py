"""VTU (VTK Unstructured Grid) reader for structured quad grids.

Reads ``.vtu`` files produced by SU2 (and similar solvers) that store
structured grids as unstructured quad meshes.  The reader reconstructs
the structured (i, j) ordering via a connectivity walk and returns
arrays shaped ``(ni, nj, 1)``.

Only raw-binary-appended encoding with UInt64 headers is currently
supported (the default output of SU2).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers._aliases import VECTOR_COMPONENTS, normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# vtu data parser
# --------------------------------------------------
def _parse_vtu_raw(fpath: Path):
    """Parse a VTU file with raw binary appended data.

    Returns
    -------
    points : ndarray, shape (n_points, 3)
    conn   : ndarray, shape (n_cells, 4)
    raw_data : dict[str, ndarray]
        PointData arrays keyed by attribute name.
    n_points, n_cells : int
    """

    # Read the entire file as bytes since the binary data is mixed with XML
    raw = fpath.read_bytes()

    # Locate the binary blob after <AppendedData encoding="raw">_
    idx = raw.find(b"<AppendedData")

    # error check: if appended data section is not found file does not contain any data we can read
    if idx == -1:
        raise ValueError(f"no <AppendedData> section found in {fpath}")

    # detect start of binary data - end of header should look something like this: <AppendedData encoding="raw">
    end_tag = raw.find(b">", idx)
    underscore = raw.find(b"_", end_tag)
    binary_start = underscore + 1

    # parse xml header (everything before binary data) to extract metadata
    xml_text = raw[:idx].decode("ascii") + "</VTKFile>"
    root = ET.fromstring(xml_text)
    piece = root.find(".//Piece")
    n_points = int(piece.get("NumberOfPoints"))
    n_cells = int(piece.get("NumberOfCells"))
    logger.debug("VTU header: %d points, %d cells", n_points, n_cells)

    # Map VTK data types to numpy dtypes
    dtype_map = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "UInt8": "<u1", "UInt64": "<u8"}

    def _read_array(da, n_items):
        offset = int(da.get("offset"))
        dt = dtype_map[da.get("type")]
        ncomp = int(da.get("NumberOfComponents", "1"))
        # Skip UInt64 length header (8 bytes)
        arr = np.frombuffer(
            raw, dtype=dt, count=n_items * ncomp, offset=binary_start + offset + 8
        )
        if ncomp > 1:
            arr = arr.reshape(n_items, ncomp)
        return arr

    points = _read_array(piece.find(".//Points/DataArray"), n_points)
    conn = _read_array(
        piece.find(".//Cells/DataArray[@Name='connectivity']"), n_cells * 4
    ).reshape(n_cells, 4)

    raw_data = {}
    for da in piece.findall(".//PointData/DataArray"):
        raw_data[da.get("Name")] = _read_array(da, n_points)

    return points, conn, raw_data, n_points, n_cells


# --------------------------------------------------
# attempt structured grid reconstruction from unstructured quads
# --------------------------------------------------
def _build_adjacency(conn, n_cells):
    """Build cell adjacency from shared quad edges.

    Returns list of dicts: neighbors[c][edge_idx] = (neighbor_cell, neighbor_edge_idx).
    """
    edge_to_cells = defaultdict(list)
    for c in range(n_cells):
        n = conn[c]
        for e_idx, edge in enumerate(
            [
                frozenset((n[0], n[1])),
                frozenset((n[1], n[2])),
                frozenset((n[2], n[3])),
                frozenset((n[3], n[0])),
            ]
        ):
            edge_to_cells[edge].append((c, e_idx))

    neighbors = [dict() for _ in range(n_cells)]
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            (c1, e1), (c2, e2) = cells
            neighbors[c1][e1] = (c2, e2)
            neighbors[c2][e2] = (c1, e1)
    return neighbors


def _walk(start_cell, start_edge, neighbors):
    """Walk from *start_cell* through *start_edge*, continuing straight."""
    path = [start_cell]
    current = start_cell
    edge = start_edge
    while edge in neighbors[current]:
        next_cell, entry_edge = neighbors[current][edge]
        path.append(next_cell)
        edge = (entry_edge + 2) % 4
        current = next_cell
    return path


def _reconstruct_cells(conn, n_cells, neighbors, points):
    """Recover structured cell ordering from unstructured quad adjacency.

    Returns cell_grid (ni, nj), ni, nj.
    Convention: i = longer direction, j = shorter direction.
    """
    corners = [c for c in range(n_cells) if len(neighbors[c]) == 2]
    if len(corners) != 4:
        raise ValueError(
            f"expected 4 corner cells (structured quad grid), found {len(corners)}"
        )

    # Start at the corner with the smallest x-centroid
    centroids = np.array([points[conn[c]].mean(axis=0) for c in corners])
    start = corners[int(np.argmin(centroids[:, 0]))]
    logger.debug("start cell %d, centroid %s", start, points[conn[start]].mean(axis=0))

    # Walk both edges from the corner — shorter = j, longer = i
    e1, e2 = sorted(neighbors[start].keys())
    path1 = _walk(start, e1, neighbors)
    path2 = _walk(start, e2, neighbors)

    if len(path1) <= len(path2):
        j_edge, i_edge = e1, e2
        j_path = path1
    else:
        j_edge, i_edge = e2, e1
        j_path = path2

    nj = len(j_path)

    # For each cell in j-column, find the perpendicular (i) edge
    perp_edges = [i_edge]
    current = start
    walk_edge = j_edge
    for k in range(1, nj):
        next_cell, entry_edge = neighbors[current][walk_edge]
        walk_next = (entry_edge + 2) % 4
        cand1 = (entry_edge + 1) % 4
        cand2 = (entry_edge + 3) % 4
        best = cand1
        for cand in (cand1, cand2):
            if cand in neighbors[next_cell]:
                if len(_walk(next_cell, cand, neighbors)) > nj:
                    best = cand
                    break
        perp_edges.append(best)
        walk_edge = walk_next
        current = next_cell

    # Walk i-direction from each j-cell
    columns = [_walk(c, perp_edges[j], neighbors) for j, c in enumerate(j_path)]
    ni = len(columns[0])

    cell_grid = np.array(columns, dtype=np.intp).T  # (ni, nj)
    logger.info("structured cell grid: ni=%d, nj=%d (%d cells)", ni, nj, ni * nj)
    return cell_grid, ni, nj


# ── node extraction ───────────────────────────────────────────────────────────


def _extract_nodes(cell_grid, conn, points):
    """Propagate node assignments to build a structured node grid.

    Returns node_grid of shape (ni+1, nj+1).
    """
    ni, nj = cell_grid.shape
    node_grid = np.full((ni + 1, nj + 1), -1, dtype=np.intp)

    def cns(i, j):
        return set(conn[cell_grid[i, j]])

    # Seed from cell (0, 0)
    c00 = cns(0, 0)
    right = c00 & cns(1, 0)
    top = c00 & cns(0, 1)

    node_grid[0, 0] = (c00 - right - top).pop()
    node_grid[1, 0] = (right - top).pop()
    node_grid[0, 1] = (top - right).pop()
    node_grid[1, 1] = (right & top).pop()

    # Propagate along j=0 (i-direction)
    for i in range(1, ni):
        cur = cns(i, 0)
        remaining = cur - {node_grid[i, 0], node_grid[i, 1]}
        next_right = cur & cns(i + 1, 0) if i < ni - 1 else remaining
        top_cur = cur & cns(i, 1) if nj > 1 else set()
        br = next_right - top_cur
        tr = next_right & top_cur
        if br and tr:
            node_grid[i + 1, 0] = br.pop()
            node_grid[i + 1, 1] = tr.pop()
        elif len(remaining) == 2:
            r = list(remaining)
            if points[r[0]][1] < points[r[1]][1]:
                node_grid[i + 1, 0], node_grid[i + 1, 1] = r[0], r[1]
            else:
                node_grid[i + 1, 0], node_grid[i + 1, 1] = r[1], r[0]

    # Propagate in j-direction layer by layer
    for j in range(1, nj):
        for i in range(ni):
            cur = cns(i, j)
            known_bl, known_br = node_grid[i, j], node_grid[i + 1, j]
            remaining = cur - {n for n in (known_bl, known_br) if n != -1}

            above = cur & cns(i, j + 1) if j < nj - 1 else remaining
            right = cur & cns(i + 1, j) if i < ni - 1 else set()

            tl = above - right if right else set()
            tr = above & right if right else set()

            if tl and tr:
                node_grid[i, j + 1] = tl.pop()
                node_grid[i + 1, j + 1] = tr.pop()
            elif len(remaining) == 2:
                r = list(remaining)
                if i > 0 and node_grid[i, j + 1] != -1:
                    node_grid[i + 1, j + 1] = (remaining - {node_grid[i, j + 1]}).pop()
                else:
                    if points[r[0]][0] < points[r[1]][0]:
                        node_grid[i, j + 1], node_grid[i + 1, j + 1] = r[0], r[1]
                    else:
                        node_grid[i, j + 1], node_grid[i + 1, j + 1] = r[1], r[0]

    filled = np.sum(node_grid != -1)
    total = (ni + 1) * (nj + 1)
    if filled < total:
        raise ValueError(f"node grid incomplete: {filled}/{total} filled")

    return node_grid


# ── public API ────────────────────────────────────────────────────────────────


def read_vtu(
    fpath: str | Path,
) -> Dataset:
    """Read a VTU file containing a structured quad grid.

    The VTU file is assumed to contain an all-quad unstructured mesh
    (as produced by SU2) that represents a structured grid.  The
    structured connectivity is recovered via a topological walk.

    Args:
        fpath: Path to the ``.vtu`` file.

    Returns:
        `Dataset` with a `StructuredGrid` of shape ``(ni, nj, 1)``.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(fpath)

    logger.info("read_vtu: %s", fpath)

    # Parse the raw VTU data
    points, conn, raw_data, n_points, n_cells = _parse_vtu_raw(fpath)

    # Build adjacency and reconstruct structured ordering
    neighbors = _build_adjacency(conn, n_cells)
    cell_grid, ni, nj = _reconstruct_cells(conn, n_cells, neighbors, points)

    # Extract structured node grid.  Orientation (j=0 wall, +i streamwise)
    # is now applied centrally by `cfd_io.convert_mod.read_file` via
    # `cfd_io.orient.canonicalize_dataset`, so the reader just returns the
    # grid in the order produced by the topology walk.
    node_grid = _extract_nodes(cell_grid, conn, points)

    ni_nodes, nj_nodes = node_grid.shape

    # Build grid dict: (ni+1, nj+1, 1)
    grid: dict[str, np.ndarray] = {
        "x": points[node_grid, 0][:, :, np.newaxis],
        "y": points[node_grid, 1][:, :, np.newaxis],
        "z": points[node_grid, 2][:, :, np.newaxis],
    }

    # Build flow dict
    flow: dict[str, np.ndarray] = {}

    # loop over all raw data arrays and assign to flow dict with shape (ni+1, nj+1, 1)
    for var_name, arr in raw_data.items():

        # normalize variable name -> rename according to _aliase.py scheme and convert to lowercase
        var_name_normalized = normalize(var_name)
        if arr.ndim == 1:
            flow[var_name_normalized] = arr[node_grid][:, :, np.newaxis]
        else:
            # Vector quantity — split into components
            names = VECTOR_COMPONENTS.get(
                var_name_normalized, (f"{var_name_normalized}_x", f"{var_name_normalized}_y", f"{var_name_normalized}_z")
            )
            for k in range(arr.shape[1]):
                flow[names[k]] = arr[node_grid, k][:, :, np.newaxis]

    attrs: dict[str, Any] = {
        "format": "vtu",
        "ni": ni_nodes,
        "nj": nj_nodes,
        "nk": 1,
        "n_vars": len(flow),
    }

    logger.info(
        "  grid: (%d, %d, 1), %d flow variables",
        ni_nodes, nj_nodes, len(flow),
    )

    return Dataset(
        grid=StructuredGrid(grid["x"], grid["y"], grid["z"]),
        flow={k: Field(v) for k, v in flow.items()},
        attrs=attrs,
    )
