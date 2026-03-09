"""Plot3D binary (Fortran unformatted) grid reader.

Reads single-block Plot3D ``.x`` grid files written with
``FORM='UNFORMATTED'``.  2-D and 3-D grids are supported.

Binary layout::

    record 1:  nblocks            (1 int32)         -- optional
    record 2:  ni, nj [, nk]      (2 or 3 int32)
    record 3:  x, y [, z]         (all reals, one record)

Only the first block is read for multi-block files.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cfd_io.readers._plot3d_common import unpack_coordinates
from cfd_io.readers.fortran_binary_sequential import FortranBinaryReader

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# read a Fortran unformatted binary Plot3D grid file
def read_plot3d_grid_binary(
    fpath: str | Path,
) -> dict[str, np.ndarray]:
    """Read a Fortran unformatted binary Plot3D grid file.

    Args:
        fpath: Path to the binary ``.x`` file.

    Returns:
        Grid dict ``{"x": (ni, nj, nk), "y": ..., "z": ...}``.
        For 2-D files *z* is omitted and *nk* = 1.
    """
    fpath = Path(fpath)

    with FortranBinaryReader(fpath) as reader:

        # first record -- either nblocks (1 int) or dimensions (2-3 ints)
        first_ints = reader.read_ints()

        if first_ints.size == 1:
            # nblocks record present -> dimensions are in the next record
            dim_ints = reader.read_ints()
        else:
            # no nblocks record -> first record IS the dimensions
            dim_ints = first_ints

        # parse 2-D or 3-D grid dimensions from the dimension record
        if dim_ints.size == 2:
            ni, nj = int(dim_ints[0]), int(dim_ints[1])
            nk = 1
            is_3d = False
        elif dim_ints.size == 3:
            ni, nj, nk = int(dim_ints[0]), int(dim_ints[1]), int(dim_ints[2])
            is_3d = True
        else:
            raise ValueError(
                f"unexpected dimension record size: {dim_ints.size} "
                f"(expected 2 or 3)"
            )

        logger.debug("  binary dims: ni=%d, nj=%d, nk=%d", ni, nj, nk)

        # coordinate data -- all x, all y, [all z] packed in one Fortran record
        n_coords = 3 if is_3d else 2
        total = ni * nj * nk * n_coords
        coord_data = reader.read_reals(expected_count=total)

    # unpack flat coordinate vector into separate x, y, z arrays
    return unpack_coordinates(coord_data, ni, nj, nk, is_3d)
