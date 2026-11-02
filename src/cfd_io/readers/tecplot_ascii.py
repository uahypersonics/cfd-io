"""Tecplot ASCII (``.dat``) reader.

Reads structured-grid Tecplot ASCII files in both POINT and BLOCK
data packing.  2-D and 3-D zones are supported.  Only the first zone
is read (multi-zone files are not yet supported).

FEQUADRILATERAL zones that hold a structured grid (as written by SU2
and CFD++) are read too: the structured (i, j) ordering is recovered
from the quad connectivity, and the arrays are ``(ni, nj, 1)``.

Expected header format::

    TITLE = "..."
    VARIABLES = "x", "y", "uvel", "vvel", "pres"
    ZONE T="Zone 1", I=100, J=50, K=1, F=POINT

or equivalently ``DATAPACKING=POINT`` / ``DATAPACKING=BLOCK``.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers._aliases import GRID_NAMES, normalize
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
# public API to read Tecplot ASCII files
# --------------------------------------------------
def read_tecplot_ascii(
    fpath: str | Path,
) -> Dataset:
    """Read a Tecplot ASCII ``.dat`` file.

    Args:
        fpath: Path to the ``.dat`` file.

    Returns:
        `Dataset` with a `StructuredGrid`.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        ValueError: If the header cannot be parsed.
    """

    # convert input file to path object
    fpath = Path(fpath)

    # check that the file exists before trying to read
    if not fpath.exists():
        raise FileNotFoundError(f"Tecplot file not found: {fpath}")

    # debug output for devs
    logger.debug("read_tecplot_ascii: %s", fpath)

    # read all lines from the file at once
    with open(fpath) as fobj:
        lines = fobj.readlines()

    # parse header to extract title, variable names, zone dimensions, and where data starts
    title, var_names, zone_info, data_start = _parse_header(lines)

    # store dimensions and packing info for later use
    ni = zone_info["i"]
    nj = zone_info["j"]
    nk = zone_info.get("k", 1)
    packing = zone_info.get("packing", "point").lower()
    n_vars = len(var_names)
    zone_type = zone_info["zone_type"]

    # check: finite-element zones are read as structured grids, which needs quads
    # with every variable at the nodes
    if zone_type != "ordered":
        if zone_type != "fequadrilateral":
            raise ValueError(
                f"unsupported Tecplot zone type {zone_type!r}: only ordered (I, J, K) "
                "and FEQUADRILATERAL zones can be read"
            )
        if zone_info["cell_centered"]:
            raise ValueError(
                "cell-centered variables (VARLOCATION=...CELLCENTERED) in "
                "finite-element zones aren't supported yet"
            )

    # debug output for devs
    logger.debug(
        "  zone: %s, %dx%dx%d, %d vars, packing=%s",
        zone_type, ni, nj, nk, n_vars, packing,
    )

    # warn if the file contains additional zones (only the first is read)
    for line in lines[data_start:]:
        stripped = line.strip().upper()
        if stripped.startswith("ZONE"):
            logger.warning(
                "multi-zone file detected; only the first zone is read: %s",
                fpath,
            )
            break

    # read numeric data (after the header -> indicated by data_start)
    values = _read_data_values(lines, data_start)

    # finite-element quads: recover the structured grid from the connectivity
    if zone_type == "fequadrilateral":
        arrays = _unpack_fe_quad(
            values,
            var_names,
            zone_info["nodes"],
            zone_info["elements"],
            packing,
        )
    else:
        # compute expcected number of values based on dimensions and variable count
        n_expected = ni * nj * nk * n_vars

        # check that we have enough data values to fill the expected arrays
        if len(values) < n_expected:
            raise ValueError(
                f"not enough data: expected {n_expected} values, got {len(values)}"
            )

        # reshape into per-variable arrays (unpack depends on whether data is POINT or BLOCK format)
        if packing == "block":
            arrays = _unpack_block(values, var_names, ni, nj, nk)
        else:
            arrays = _unpack_point(values, var_names, ni, nj, nk)

    # split into grid and flow -> first initialize empty dicts
    grid: dict[str, np.ndarray] = {}
    flow: dict[str, np.ndarray] = {}

    # assign variables to grid vs flow based on name (e.g. "x", "y", "z" go to grid)
    # GRID_NAMES is provided in _aliases.py
    for var_name, arr in arrays.items():
        var_name_normalized = normalize(var_name)
        if var_name_normalized.lower() in GRID_NAMES:
            grid[var_name_normalized.lower()] = arr
        else:
            flow[var_name_normalized] = arr

    # store title as metadata if present
    attrs: dict[str, Any] = {}
    if title:
        attrs["title"] = title

    # build structured grid only when x and y coordinates are present
    if "x" in grid and "y" in grid:
        grid_obj = StructuredGrid(
            grid["x"],
            grid["y"],
            grid.get("z", np.zeros_like(grid["x"])),
        )
    else:
        # when coordinate variables are absent, keep all variables as flow
        # so non-CFD Tecplot tables (e.g. LST output) can still be loaded
        for var_name, arr in grid.items():
            flow[var_name] = arr

        grid_obj = None
        logger.warning(
            "read_tecplot_ascii: missing x/y coordinates; returning flow-only dataset for %s",
            fpath,
        )

    logger.info(
        "read_tecplot_ascii: grid(%d) + flow(%d) from %s",
        len(grid), len(flow), fpath,
    )

    return Dataset(
        grid=grid_obj,
        flow={k: Field(v) for k, v in flow.items()},
        attrs=attrs,
    )


# --------------------------------------------------
# parse through header
# --------------------------------------------------
def _parse_header(
    lines: list[str],
) -> tuple[str, list[str], dict[str, Any], int]:
    """Parse the Tecplot ASCII header.

    Args:
        lines: All lines read from the Tecplot file.

    Returns:
        ``(title, var_names, zone_info, data_start_line_index)``

        *zone_info* contains keys ``"i"``, ``"j"``, ``"k"`` (int),
        and ``"packing"`` (``"point"`` or ``"block"``).
    """

    # initialize outputs
    title = ""
    var_names: list[str] = []
    zone_info: dict[str, Any] = {}
    line_idx = 0

    # walk through lines looking for TITLE, VARIABLES, and ZONE keywords
    while line_idx < len(lines):
        stripped = lines[line_idx].strip()

        # skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            line_idx += 1
            continue

        upper = stripped.upper()

        # find TITLE line
        if upper.startswith("TITLE"):
            title = _extract_quoted(stripped)
            line_idx += 1
            continue

        # find VARIABLES line (may span multiple lines)
        if upper.startswith("VARIABLES"):
            var_names, line_idx = _parse_variables(lines, line_idx)
            continue

        # find ZONE line

        if upper.startswith("ZONE"):
            zone_info, line_idx = _parse_zone(lines, line_idx)
            break

        # if we hit numeric data before a ZONE line, stop
        if _looks_numeric(stripped):
            break

        line_idx += 1

    if not var_names:
        raise ValueError("no VARIABLES line found in header")

    # check: the zone says how many points it holds (I=... or N=...); without
    # either, the data can't be split into points
    if "i" not in zone_info and "nodes" not in zone_info:
        raise ValueError(
            "ZONE header has no point count: expected I=, J=, K= (ordered zone) "
            "or N=, E= (finite-element zone)"
        )

    # apply defaults for any missing zone parameters
    zone_info.setdefault("i", 1)
    zone_info.setdefault("j", 1)
    zone_info.setdefault("k", 1)
    zone_info.setdefault("packing", "point")
    zone_info.setdefault("zone_type", "ordered")

    return title, var_names, zone_info, line_idx


# extract text between the first pair of double quotes
def _extract_quoted(s: str) -> str:
    """Extract text between the first pair of double quotes."""
    m = re.search(r'"([^"]*)"', s)
    return m.group(1) if m else ""


# parse VARIABLES = "x", "y", ... (may span multiple lines)
def _parse_variables(
    lines: list[str],
    start: int,
) -> tuple[list[str], int]:
    """Parse VARIABLES = "x", "y", ... (may span multiple lines)."""

    # join lines until we hit ZONE or numeric data (VARIABLES can wrap)
    combined = ""
    idx = start

    while idx < len(lines):
        stripped = lines[idx].strip()
        combined += " " + stripped
        idx += 1

        # peek at next line -- stop if it starts ZONE or looks numeric
        if idx < len(lines):
            next_upper = lines[idx].strip().upper()
            if next_upper.startswith("ZONE") or _looks_numeric(lines[idx].strip()):
                break
            # also stop if the next line doesn't look like a continuation
            if not re.match(r'^[",\s]', lines[idx].strip()):
                # check if it could be a continued quoted name
                if not lines[idx].strip().startswith('"'):
                    break

    # try to extract all quoted names first (e.g. "x", "y", "uvel")
    names = re.findall(r'"([^"]*)"', combined)

    if not names:
        # fall back to unquoted comma-separated: VARIABLES = x, y, u
        after_eq = combined.split("=", 1)[-1]
        names = [n.strip() for n in after_eq.split(",") if n.strip()]

    return names, idx


# parse ZONE line(s) into a dict of zone parameters
def _parse_zone(
    lines: list[str],
    start: int,
) -> tuple[dict[str, Any], int]:
    """Parse ZONE line(s) into a dict of zone parameters."""

    # the ZONE line may wrap across multiple lines -- join until we hit data
    combined = ""
    idx = start

    while idx < len(lines):
        combined += " " + lines[idx].strip()
        idx += 1

        # peek: is next line numeric data?
        if idx < len(lines) and _looks_numeric(lines[idx].strip()):
            break
        # or is next line another keyword?
        if idx < len(lines):
            next_upper = lines[idx].strip().upper()
            if next_upper.startswith(("ZONE", "VARIABLES", "TITLE", "TEXT")):
                break
        # the zone line usually fits on one line, but be safe
        # stop if we already have dimensions and line doesn't end with comma
        if re.search(r"[IJK]\s*=", combined, re.IGNORECASE):
            if not combined.rstrip().endswith(","):
                break

    info: dict[str, Any] = {}

    # extract I, J, K dimension values from the combined zone string
    for dim in ("I", "J", "K"):
        m = re.search(rf"\b{dim}\s*=\s*(\d+)", combined, re.IGNORECASE)
        if m:
            info[dim.lower()] = int(m.group(1))

    # extract node and element counts of finite-element zones (N=/Nodes=, E=/Elements=);
    # whole words only, so e.g. "SOLUTIONTIME= 0.0e+00" doesn't count as E=
    m_n = re.search(r"\b(?:N|Nodes)\s*=\s*(\d+)", combined, re.IGNORECASE)
    m_e = re.search(r"\b(?:E|Elements)\s*=\s*(\d+)", combined, re.IGNORECASE)
    if m_n and m_e:
        info["nodes"] = int(m_n.group(1))
        info["elements"] = int(m_e.group(1))

    # extract the element type: ZONETYPE=FEQUADRILATERAL or ET=QUADRILATERAL
    m_zt = re.search(r"ZONETYPE\s*=\s*(\w+)", combined, re.IGNORECASE)
    m_et = re.search(r"\bET\s*=\s*(\w+)", combined, re.IGNORECASE)
    if m_zt:
        info["zone_type"] = m_zt.group(1).lower()
    elif m_et:
        info["zone_type"] = "fe" + m_et.group(1).lower()
    else:
        info["zone_type"] = "ordered"

    # check for cell-centered variables (VARLOCATION=([4-12]=CELLCENTERED))
    m_vl = re.search(r"VARLOCATION\s*=\s*\((.*)\)", combined, re.IGNORECASE)
    info["cell_centered"] = bool(m_vl and "CELLCENTERED" in m_vl.group(1).upper())

    # extract data packing format: F=POINT or DATAPACKING=BLOCK
    m_f = re.search(r"\bF\s*=\s*(\w+)", combined, re.IGNORECASE)
    m_dp = re.search(r"DATAPACKING\s*=\s*(\w+)", combined, re.IGNORECASE)

    if m_dp:
        info["packing"] = m_dp.group(1).lower()
    elif m_f:
        info["packing"] = m_f.group(1).lower()
    else:
        # default to POINT if not specified
        info["packing"] = "point"

    # older finite-element headers combine packing and zone type: F=FEPOINT, F=FEBLOCK
    if info["packing"] in ("fepoint", "feblock"):
        info["packing"] = info["packing"].removeprefix("fe")

    # extract zone title if present (e.g. T="Zone 1")
    m_t = re.search(r'T\s*=\s*"([^"]*)"', combined, re.IGNORECASE)
    if m_t:
        info["zone_title"] = m_t.group(1)

    return info, idx


# quick check: does the string start with a number or sign?
def _looks_numeric(s: str) -> bool:
    """Return True if the string starts with a number or sign."""
    s = s.lstrip()
    if not s:
        return False
    return bool(re.match(r"^[+\-]?[\d.]", s))


# --------------------------------------------------
# data unpacking
# --------------------------------------------------
def _read_data_values(lines: list[str], start: int) -> np.ndarray:
    """Read all numeric values from lines[start:]."""

    values: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()

        # skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # parse each whitespace-delimited token as a float
        for tok in stripped.split():
            try:
                values.append(float(tok))
            except ValueError:
                # stop at non-numeric tokens (e.g. next ZONE header)
                break

    return np.array(values, dtype=np.float64)


# unpack POINT-format data: each row has all variables for one point
def _unpack_point(
    values: np.ndarray,
    var_names: list[str],
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, np.ndarray]:
    """Unpack POINT-format data: each row has all variables for one point.

    Point ordering is i-fastest, then j, then k.
    """
    n_vars = len(var_names)
    n_points = ni * nj * nk

    # reshape flat values into (n_points, n_vars) table
    raw = values[: n_points * n_vars].reshape((n_points, n_vars))

    # extract each column as a 3-D array in Fortran order (i-fastest)
    arrays: dict[str, np.ndarray] = {}
    for col, name in enumerate(var_names):
        flat = raw[:, col]
        # reshape from point order (i-fastest, j, k) to (ni, nj, nk)
        arr = flat.reshape((ni, nj, nk), order="F")
        arrays[name] = arr

    return arrays


# unpack BLOCK-format data: all values for one variable, then the next
def _unpack_block(
    values: np.ndarray,
    var_names: list[str],
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, np.ndarray]:
    """Unpack BLOCK-format data: all values for one variable, then the next.

    Within each variable, ordering is i-fastest, then j, then k.
    """
    n_points = ni * nj * nk
    offset = 0

    # each variable occupies n_points consecutive values
    arrays: dict[str, np.ndarray] = {}
    for name in var_names:
        flat = values[offset : offset + n_points]
        # reshape from flat to 3-D in Fortran order (i-fastest)
        arr = flat.reshape((ni, nj, nk), order="F")
        arrays[name] = arr
        offset += n_points

    return arrays


# unpack a FEQUADRILATERAL zone that holds a structured grid
def _unpack_fe_quad(
    values: np.ndarray,
    var_names: list[str],
    n_nodes: int,
    n_elements: int,
    packing: str,
) -> dict[str, np.ndarray]:
    """Unpack a FEQUADRILATERAL zone into structured ``(ni, nj, 1)`` arrays.

    Solvers such as SU2 and CFD++ write structured grids as finite-element
    quad zones: the nodal values (POINT or BLOCK packed) followed by the
    connectivity, four 1-based node indices per element.  The structured
    (i, j) ordering is recovered from the connectivity alone, the same way
    the VTU reader does it.

    Args:
        values: All numeric values of the zone, in file order.
        var_names: Variable names from the header.
        n_nodes: Number of nodes (N=).
        n_elements: Number of quad elements (E=).
        packing: ``"point"`` or ``"block"``.

    Returns:
        Per-variable arrays of shape ``(ni, nj, 1)``.

    Raises:
        ValueError: If data is missing, x/y are absent, or the quads don't
            form one structured block.
    """

    # check that we have the nodal values and the connectivity
    n_vars = len(var_names)
    n_nodal = n_nodes * n_vars
    n_expected = n_nodal + 4 * n_elements
    if len(values) < n_expected:
        raise ValueError(
            f"not enough data for {n_nodes} nodes and {n_elements} quads: "
            f"expected {n_expected} values, got {len(values)}"
        )

    # nodal values per variable (POINT: one row per node; BLOCK: one variable after another)
    nodal = values[:n_nodal]
    if packing == "block":
        table = nodal.reshape((n_vars, n_nodes)).T
    else:
        table = nodal.reshape((n_nodes, n_vars))
    columns = {name: table[:, col] for col, name in enumerate(var_names)}

    # connectivity, converted from 1-based Tecplot indices to 0-based
    conn = values[n_nodal:n_expected].astype(np.intp).reshape((n_elements, 4)) - 1

    # node coordinates for the reconstruction (z = 0 for planar data)
    coords: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        grid_name = normalize(name).lower()
        if grid_name in GRID_NAMES:
            coords[grid_name] = column
    if "x" not in coords or "y" not in coords:
        raise ValueError("finite-element zone without x and y coordinates")
    z = coords.get("z", np.zeros(n_nodes))
    points = np.column_stack([coords["x"], coords["y"], z])

    # recover the structured cell and node ordering from the quad adjacency
    neighbors = _build_adjacency(conn, n_elements)
    cell_grid, _, _ = _reconstruct_cells(conn, n_elements, neighbors, points)
    node_grid = _extract_nodes(cell_grid, conn, points)

    # check: every node belongs to the structured block
    if node_grid.size != n_nodes:
        raise ValueError(
            f"the quads don't form one structured block: {node_grid.size} of "
            f"{n_nodes} nodes are on the reconstructed grid"
        )

    # debug output for devs
    logger.debug("  structured grid from quads: %s", node_grid.shape)

    # gather each variable onto the structured node grid
    arrays: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        arrays[name] = column[node_grid][:, :, np.newaxis]

    return arrays
