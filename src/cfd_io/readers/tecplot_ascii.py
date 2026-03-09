"""Tecplot ASCII (``.dat``) reader.

Reads structured-grid Tecplot ASCII files in both POINT and BLOCK
data packing.  2-D and 3-D zones are supported.  Only the first zone
is read (multi-zone files are not yet supported).

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

from cfd_io.readers._aliases import GRID_NAMES, normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API to read Tecplot ASCII files
# --------------------------------------------------
def read_tecplot_ascii(
    fpath: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read a Tecplot ASCII ``.dat`` file.

    Args:
        fpath: Path to the ``.dat`` file.

    Returns:
        Tuple of ``(grid, flow, attrs)`` where:

        - **grid** -- ``{"x": (ni, nj, nk), "y": ..., ...}``
        - **flow** -- ``{"uvel": (ni, nj, nk), ...}``
        - **attrs** -- ``{"title": str}`` if a title was present

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

    # debug output for devs
    logger.debug(
        "  zone: %dx%dx%d, %d vars, packing=%s",
        ni, nj, nk, n_vars, packing,
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

    logger.info(
        "read_tecplot_ascii: grid(%d) + flow(%d) from %s",
        len(grid), len(flow), fpath,
    )

    # return grid, flow, and any attributes (e.g. title) as a tuple
    return grid, flow, attrs


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

    # apply defaults for any missing zone parameters
    zone_info.setdefault("i", 1)
    zone_info.setdefault("j", 1)
    zone_info.setdefault("k", 1)
    zone_info.setdefault("packing", "point")

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
        m = re.search(rf"{dim}\s*=\s*(\d+)", combined, re.IGNORECASE)
        if m:
            info[dim.lower()] = int(m.group(1))

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
