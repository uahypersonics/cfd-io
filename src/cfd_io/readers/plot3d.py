"""Plot3D structured grid reader -- auto-detecting dispatcher.

Detects whether a ``.x`` file is ASCII or Fortran unformatted binary
and delegates to the appropriate reader.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# plot 3d reader dispatcher
# --------------------------------------------------
def read_plot3d(
    fpath: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read a Plot3D grid file (ASCII or binary, 2-D or 3-D).

    The file format is auto-detected by inspecting the first 4 bytes.

    Args:
        fpath: Path to the ``.x`` file.

    Returns:
        Tuple of ``(grid, flow, attrs)``:

        - **grid** -- ``{"x": (nx, ny, nz), "y": ..., "z": ...}``
        - **flow** -- always ``{}`` (Plot3D grid files have no flow data)
        - **attrs** -- empty dict
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    # raise error if file does not exist
    if not fpath.exists():
        raise FileNotFoundError(f"Plot3D file not found: {fpath}")

    # auto-detect binary vs ASCII by inspecting the first bytes
    if _is_binary(fpath):
        from cfd_io.readers.plot3d_grid_binary import read_plot3d_grid_binary

        logger.debug("read_plot3d: %s  (binary)", fpath)
        grid = read_plot3d_grid_binary(fpath)
    else:
        from cfd_io.readers.plot3d_grid_ascii import read_plot3d_grid_ascii

        logger.debug("read_plot3d: %s  (ASCII)", fpath)
        grid = read_plot3d_grid_ascii(fpath)

    # Plot3D grid files contain no flow data or attributes
    flow: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}

    n_vars = len(grid)
    first = next(iter(grid.values()))
    logger.info(
        "read_plot3d: %d vars, shape=%s from %s",
        n_vars, first.shape, fpath,
    )

    return grid, flow, attrs


# --------------------------------------------------
# format detection
# --------------------------------------------------

# heuristic check: plausible Fortran record marker in the first 4 bytes
def _is_binary(fpath: Path) -> bool:
    """Heuristic: if the first 4 bytes are a plausible Fortran record
    marker (4, 8, or 12 = 1, 2, or 3 int32 values), treat as binary.
    """
    with open(fpath, "rb") as fobj:
        raw = fobj.read(4)

    if len(raw) < 4:
        return False

    # Fortran record marker for nblocks=1 is 4 bytes, for 2 ints is 8, for 3 ints is 12
    marker = struct.unpack("<i", raw)[0]
    return marker in (4, 8, 12)
