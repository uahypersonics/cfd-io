"""Plot3D ASCII grid writer.

Writes single-block Plot3D ``.x`` grid files in plain-text format.
2-D and 3-D grids are supported.

ASCII layout::

    1
    ni  nj  [nk]
    x(i,j[,k])  y(...)  [z(...)]
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# write a grid dict to an ASCII Plot3D file
def write_plot3d_grid_ascii(
    fpath: str | Path,
    grid: dict[str, np.ndarray],
) -> None:
    """Write a grid dict to an ASCII Plot3D file.

    Args:
        fpath: Output file path.
        grid: Grid arrays ``{"x": (ni, nj, nk), "y": ..., "z": ...}``.
            If ``"z"`` is absent the file is written as 2-D.
    """
    # convert to Path object and determine dimensionality
    fpath = Path(fpath)
    is_3d = "z" in grid
    ni, nj, nk = grid["x"].shape

    # build ordered list of coordinate arrays: x, y, [z]
    coord_arrays = [grid["x"], grid["y"]]
    if is_3d:
        coord_arrays.append(grid["z"])

    with open(fpath, "w") as fobj:

        # line 1: number of blocks (always 1 for single-block output)
        fobj.write("1\n")

        # line 2: grid dimensions (2 for 2-D, 3 for 3-D)
        if is_3d:
            fobj.write(f"{ni} {nj} {nk}\n")
        else:
            fobj.write(f"{ni} {nj}\n")

        # coordinate data -- Fortran column-major order, 5 values per line
        # each coordinate array is flattened in F-order and written sequentially
        for arr in coord_arrays:
            flat = arr.ravel(order="F")
            for row_start in range(0, len(flat), 5):
                chunk = flat[row_start : row_start + 5]
                line = " ".join(f"{v:20.12E}" for v in chunk)
                fobj.write(line + "\n")

    # debug output for devs
    logger.debug("write_plot3d_grid_ascii: %s (%dx%dx%d)", fpath, ni, nj, nk)
