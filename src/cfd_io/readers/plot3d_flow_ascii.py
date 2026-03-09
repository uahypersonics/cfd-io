"""Plot3D ASCII solution reader.

Reads single-block Plot3D ``.q`` solution files in plain-text format.
2-D and 3-D grids are supported.

ASCII layout::

    line 1:  nblocks                                -- optional
    line 2:  ni  nj [nk]
    line 3:  mach  alpha  re  time
    data  :  dens(i,j[,k])  then  xmom(...)  then  ymom(...)
             [then zmom(...)]  then  energy(...)

Only the first block is read for multi-block files.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.readers._plot3d_common import parse_freestream, unpack_flow

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# read an ASCII Plot3D solution file
def read_plot3d_flow_ascii(
    fpath: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read an ASCII Plot3D solution file.

    Args:
        fpath: Path to the ASCII ``.q`` file.

    Returns:
        Tuple of ``(flow, attrs)`` where:

        - **flow** -- ``{"dens": (ni,nj,nk), "xmom": ..., ...}``
        - **attrs** -- ``{"mach": float, "alpha": float, "re": float, "time": float}``

        For 2-D files *zmom* is omitted and *nk* = 1.
    """
    fpath = Path(fpath)

    # read all lines from the file at once
    with open(fpath) as fobj:
        lines = fobj.readlines()

    # skip blank lines at the top of the file
    line_idx = 0
    while line_idx < len(lines) and lines[line_idx].strip() == "":
        line_idx += 1

    # first non-blank line: either nblocks (1 token) or dimensions (2-3 tokens)
    first_tokens = lines[line_idx].split()
    line_idx += 1

    if len(first_tokens) == 1:
        # nblocks line present -> dimensions are on the next line
        dim_tokens = lines[line_idx].split()
        line_idx += 1
    else:
        # no nblocks line -> first line IS the dimensions
        dim_tokens = first_tokens

    # parse 2-D or 3-D dimensions
    if len(dim_tokens) == 2:
        ni, nj = int(dim_tokens[0]), int(dim_tokens[1])
        nk = 1
        is_3d = False
    elif len(dim_tokens) >= 3:
        ni, nj, nk = int(dim_tokens[0]), int(dim_tokens[1]), int(dim_tokens[2])
        is_3d = True
    else:
        raise ValueError(f"cannot parse dimensions from: {dim_tokens}")

    logger.debug("  ASCII dims: ni=%d, nj=%d, nk=%d", ni, nj, nk)

    # freestream conditions line: mach, alpha, re, time
    fs_tokens = lines[line_idx].split()
    line_idx += 1
    freestream = np.array([float(t) for t in fs_tokens[:4]], dtype=np.float64)
    attrs = parse_freestream(freestream)
    logger.debug("  freestream: mach=%.4f, alpha=%.2f, re=%.2e, time=%.4e",
                  attrs["mach"], attrs["alpha"], attrs["re"], attrs["time"])

    # gather all remaining numeric values from the data lines into a flat list
    values: list[float] = []
    for line in lines[line_idx:]:
        for tok in line.split():
            values.append(float(tok))

    # verify we have enough data for the expected number of flow variables
    n_vars = 5 if is_3d else 4
    expected = ni * nj * nk * n_vars
    if len(values) < expected:
        raise ValueError(
            f"not enough data: expected {expected} values, got {len(values)}"
        )

    # convert to numpy and unpack into separate flow variable arrays
    flow_data = np.array(values[:expected], dtype=np.float64)
    flow = unpack_flow(flow_data, ni, nj, nk, is_3d)

    logger.info(
        "read_plot3d_flow_ascii: %d vars, shape=(%d,%d,%d) from %s",
        len(flow), ni, nj, nk, fpath,
    )

    return flow, attrs
