"""Plot3D solution reader -- auto-detecting dispatcher.

Detects whether a ``.q`` file is ASCII or Fortran unformatted binary
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
# plot3d solution reader dispatcher
# --------------------------------------------------
def read_plot3d_flow(
    fpath: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read a Plot3D solution file (ASCII or binary, 2-D or 3-D).

    The file format is auto-detected by inspecting the first 4 bytes.

    Args:
        fpath: Path to the ``.q`` file.

    Returns:
        Tuple of ``(grid, flow, attrs)``:

        - **grid** -- always ``{}`` (``.q`` files have no grid data)
        - **flow** -- ``{"dens": (ni,nj,nk), "xmom": ..., ...}``
        - **attrs** -- ``{"mach": float, "alpha": float, "re": float, "time": float}``
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    # raise error if file does not exist
    if not fpath.exists():
        raise FileNotFoundError(f"Plot3D solution file not found: {fpath}")

    # auto-detect binary vs ASCII by inspecting the first bytes
    if _is_binary(fpath):
        from cfd_io.readers.plot3d_flow_binary import read_plot3d_flow_binary

        logger.debug("read_plot3d_flow: %s  (binary)", fpath)
        flow, attrs = read_plot3d_flow_binary(fpath)
    else:
        from cfd_io.readers.plot3d_flow_ascii import read_plot3d_flow_ascii

        logger.debug("read_plot3d_flow: %s  (ASCII)", fpath)
        flow, attrs = read_plot3d_flow_ascii(fpath)

    # .q files contain no grid data
    grid: dict[str, np.ndarray] = {}

    n_vars = len(flow)
    first = next(iter(flow.values()))
    logger.info(
        "read_plot3d_flow: %d vars, shape=%s from %s",
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
