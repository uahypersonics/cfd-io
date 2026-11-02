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
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers._aliases import VECTOR_COMPONENTS, normalize
from cfd_io.readers._structured_quads import (
    _build_adjacency,
    _extract_nodes,
    _reconstruct_cells,
)

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# vtu data parser
# --------------------------------------------------
def _parse_vtu_raw(fpath: Path):
    """Parse a VTU file with raw binary appended data.

    Returns:
        tuple: ``(points, conn, raw_data, n_points, n_cells)`` where
            ``points`` is an ndarray of shape (n_points, 3), ``conn`` is an
            ndarray of shape (n_cells, 4), ``raw_data`` is a dict of
            PointData arrays keyed by attribute name, and ``n_points`` and
            ``n_cells`` are ints.
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
