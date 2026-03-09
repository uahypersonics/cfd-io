"""Plot3D binary (Fortran unformatted) grid writer.

Writes single-block Plot3D ``.x`` grid files using Fortran sequential
unformatted I/O with record markers.  2-D and 3-D grids are supported.

Binary layout::

    record 1:  nblocks = 1        (1 int32)
    record 2:  ni, nj [, nk]      (2 or 3 int32)
    record 3:  x, y [, z]         (all reals, one record)
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cfd_io.writers.fortran_binary_sequential import FortranBinaryWriter

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# write a grid dict to a Fortran unformatted binary Plot3D file
def write_plot3d_grid_binary(
    fpath: str | Path,
    grid: dict[str, np.ndarray],
) -> None:
    """Write a grid dict to a Fortran unformatted binary Plot3D file.

    Args:
        fpath: Output file path.
        grid: Grid arrays ``{"x": (ni, nj, nk), "y": ..., "z": ...}``.
            If ``"z"`` is absent the file is written as 2-D.
    """
    # convert to Path object and determine dimensionality
    fpath = Path(fpath)
    is_3d = "z" in grid
    ni, nj, nk = grid["x"].shape

    with FortranBinaryWriter(fpath) as writer:

        # record 1: number of blocks (always 1 for single-block output)
        writer.write_ints([1])

        # record 2: grid dimensions (2 ints for 2-D, 3 ints for 3-D)
        if is_3d:
            writer.write_ints([ni, nj, nk])
        else:
            writer.write_ints([ni, nj])

        # record 3: coordinate data -- all in one record, Fortran column-major order
        # build list of coordinate arrays in order: x, y, [z]
        coord_arrays = [grid["x"], grid["y"]]
        if is_3d:
            coord_arrays.append(grid["z"])

        # flatten each coordinate array in Fortran order and concatenate into one vector
        flat_parts = []
        for arr in coord_arrays:
            arr_f64 = np.asarray(arr, dtype=np.float64)
            flat = arr_f64.ravel(order="F")
            flat_parts.append(flat)

        combined = np.concatenate(flat_parts)

        # write the combined coordinate vector as a single Fortran record
        # fortran_order=False because we already flattened in F-order above
        writer.write_array_real(combined, fortran_order=False)

    # debug output for devs
    logger.debug("write_plot3d_grid_binary: %s (%dx%dx%d)", fpath, ni, nj, nk)
