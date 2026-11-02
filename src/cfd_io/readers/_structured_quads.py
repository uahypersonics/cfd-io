"""Structured-grid reconstruction from unstructured quad meshes.

Many solvers (SU2, CFD++) store a structured grid as an unstructured
all-quad mesh.  These helpers recover the structured (i, j) ordering
from the quad connectivity alone: a topological walk from a corner cell
gives the cell grid, and the shared corners of neighboring cells give
the node grid.  Used by the VTU and Tecplot ASCII readers.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


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
        raise ValueError(f"expected 4 corner cells (structured quad grid), found {len(corners)}")

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
