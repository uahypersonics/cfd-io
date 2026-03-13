"""Read CGNS files (CFD General Notation System).

CGNS files use HDF5 as the underlying storage format, with a standardised
tree layout defined by the SIDS (Standard Interface Data Structures).

This reader handles:

- *Unstructured* zones with MIXED element connectivity (type codes
  embedded in the connectivity array) or uniform element types.
- Node-centred ``FlowSolution`` data.
- Multiple ``Elements_t`` sections per zone.

The only runtime dependency beyond the standard library is **h5py**.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cfd_io.dataset import Dataset, Field, UnstructuredGrid
from cfd_io.readers._aliases import normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CGNS element type codes
# --------------------------------------------------
# Subset relevant to surface / volume meshes.  Maps CGNS element type
# code → number of nodes per element.

CGNS_ELEMENT_NODES: dict[int, int] = {
    2: 1,   # NODE
    3: 2,   # BAR_2
    5: 3,   # TRI_3
    7: 4,   # QUAD_4
    10: 4,  # TETRA_4
    12: 5,  # PYRA_5
    14: 6,  # PENTA_6
    17: 8,  # HEXA_8
}

CGNS_MIXED = 20


# --------------------------------------------------
# helper functions
# --------------------------------------------------
def _read_data(group: h5py.Group) -> np.ndarray:
    """Read the data array from a CGNS node (stored under the key ``' data'``)."""
    return group[" data"][:]


def _read_string(group: h5py.Group) -> str:
    """Read a CGNS character-array node as a Python string."""
    raw = _read_data(group)
    return bytes(raw).decode("ascii").strip("\x00").strip()


def _iter_children(group: h5py.Group, label: str):
    """Yield ``(name, child_group)`` pairs whose CGNS *label* matches."""
    for name in group:
        child = group[name]
        if isinstance(child, h5py.Group):
            child_label = child.attrs.get("label", b"")
            if isinstance(child_label, bytes):
                child_label = child_label.decode()
            if child_label == label:
                yield name, child


# --------------------------------------------------
# element parsing
# --------------------------------------------------
def _parse_elements(
    section: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an ``Elements_t`` section.

    Returns:
        connectivity: flat int array of 0-based node indices per cell.
        offsets: int array of shape ``(n_cells + 1,)``.
        cell_types: int array of CGNS element type codes, one per cell.
    """
    elem_info = _read_data(section)  # [type_code, 0]
    etype = int(elem_info[0])

    ec_group = section["ElementConnectivity"]
    raw_conn = _read_data(ec_group)

    if etype == CGNS_MIXED:
        return _parse_mixed_elements(section, raw_conn)

    # uniform element type — all cells have the same type
    n_nodes = CGNS_ELEMENT_NODES.get(etype)
    if n_nodes is None:
        raise ValueError(f"unsupported CGNS element type code {etype}")

    # connectivity is a flat array of 1-based node indices
    n_cells = len(raw_conn) // n_nodes
    conn = raw_conn - 1  # convert to 0-based
    offsets = np.arange(0, n_cells * n_nodes + 1, n_nodes, dtype=np.int64)
    cell_types = np.full(n_cells, etype, dtype=np.int32)
    return conn, offsets, cell_types


# --------------------------------------------------
# parse MIXED element connectivity with type codes embedded
# --------------------------------------------------
def _parse_mixed_elements(
    section: h5py.Group,
    raw_conn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse MIXED element connectivity with type codes embedded.

    In CGNS MIXED format with ``ElementStartOffset``, each element's
    slice in the connectivity array is: ``[type_code, node1, node2, ...]``.
    """
    raw_offsets = _read_data(section["ElementStartOffset"])
    n_cells = len(raw_offsets) - 1

    cell_types = np.empty(n_cells, dtype=np.int32)
    # strip type codes from connectivity, collect pure node indices
    clean_parts: list[np.ndarray] = []
    clean_offsets = [0]

    for i in range(n_cells):
        start = int(raw_offsets[i])
        end = int(raw_offsets[i + 1])
        cell_types[i] = raw_conn[start]
        nodes = raw_conn[start + 1 : end]
        clean_parts.append(nodes)
        clean_offsets.append(clean_offsets[-1] + len(nodes))

    conn = np.concatenate(clean_parts) - 1  # 0-based
    offsets = np.array(clean_offsets, dtype=np.int64)
    return conn, offsets, cell_types


# --------------------------------------------------
# main reader function
# --------------------------------------------------
def read_cgns(fpath: str | Path) -> Dataset:
    """Read a CGNS/HDF5 file.

    Reads the first zone in the first base.  Returns an unstructured
    ``Dataset`` with node coordinates, element connectivity, and all
    ``FlowSolution`` variables.

    Args:
        fpath: Path to the ``.cgns`` file.

    Returns:
        `Dataset` with an `UnstructuredGrid`.
    """

    # ensure fpath is Path object
    fpath = Path(fpath)

    # make sure fpath exists and is a file
    if not fpath.is_file():
        raise FileNotFoundError(fpath)

    logger.info("read_cgns: %s", fpath)

    # open file with h5py and navigate CGNS tree structure
    with h5py.File(fpath, "r") as f:

        # locate first base
        bases = list(_iter_children(f, "CGNSBase_t"))
        if not bases:
            raise ValueError("no CGNSBase_t found in file")
        base_name, base = bases[0]
        logger.debug("  base: %s", base_name)

        # locate first zone within base
        zones = list(_iter_children(base, "Zone_t"))
        if not zones:
            raise ValueError(f"no Zone_t found in base '{base_name}'")
        zone_name, zone = zones[0]
        logger.debug("  zone: %s", zone_name)

        # zone_data: [[n_vertices], [n_cells], [n_boundary]] for unstructured
        zone_data = _read_data(zone)
        n_vertices = int(zone_data[0, 0]) if zone_data.ndim == 2 else int(zone_data[0])
        logger.info("  vertices: %d", n_vertices)

        # coordinates
        gc = dict(_iter_children(zone, "GridCoordinates_t"))
        if not gc:
            raise ValueError("no GridCoordinates_t node found")
        gc_node = list(gc.values())[0]

        coord_map = {}
        for name, child in _iter_children(gc_node, "DataArray_t"):
            coord_map[name] = _read_data(child)

        x = coord_map.get("CoordinateX", np.zeros(n_vertices))
        y = coord_map.get("CoordinateY", np.zeros(n_vertices))
        z = coord_map.get("CoordinateZ", np.zeros(n_vertices))
        points = np.column_stack([x, y, z])

        logger.debug("  points shape: %s", points.shape)

        # elements
        all_conn: list[np.ndarray] = []
        all_offsets: list[np.ndarray] = []
        all_types: list[np.ndarray] = []

        for sec_name, sec in _iter_children(zone, "Elements_t"):
            logger.debug("  element section: %s", sec_name)
            conn, offs, ctypes = _parse_elements(sec)
            # shift offsets to account for preceding sections
            if all_conn:
                conn_offset = sum(len(c) for c in all_conn)
                offs = offs + conn_offset
            all_conn.append(conn)
            all_offsets.append(offs[:-1])  # drop trailing entry (next section starts there)
            all_types.append(ctypes)

        if not all_conn:
            raise ValueError("no Elements_t sections found")

        connectivity = np.concatenate(all_conn)
        # append final offset (total connectivity length)
        offsets = np.append(np.concatenate(all_offsets), len(connectivity))
        cell_types = np.concatenate(all_types)
        n_cells = len(cell_types)
        logger.info("  cells: %d", n_cells)

        # build flow solution dict
        flow: dict[str, Field] = {}

        for fs_name, fs_node in _iter_children(zone, "FlowSolution_t"):
            logger.debug("  flow solution: %s", fs_name)

            # determine association from GridLocation if present
            association: str = "node"
            gl_nodes = list(_iter_children(fs_node, "GridLocation_t"))
            if gl_nodes:
                loc = _read_string(gl_nodes[0][1])
                if "Cell" in loc:
                    association = "cell"

            for var_name, da_node in _iter_children(fs_node, "DataArray_t"):
                canon = normalize(var_name)
                arr = _read_data(da_node)
                flow[canon] = Field(data=arr, association=association)
                logger.debug("    %s -> %s  shape=%s", var_name, canon, arr.shape)

    # build UnstructuredGrid object from parsed data
    grid = UnstructuredGrid(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
    )

    # build attributes dict with metadata about the dataset
    attrs: dict[str, Any] = {
        "format": "cgns",
        "n_vertices": n_vertices,
        "n_cells": n_cells,
        "n_vars": len(flow),
    }

    logger.info(
        "  grid: %d points, %d cells, %d flow variables",
        n_vertices,
        n_cells,
        len(flow),
    )

    return Dataset(grid=grid, flow=flow, attrs=attrs)
