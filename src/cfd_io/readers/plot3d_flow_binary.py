"""Plot3D binary (Fortran unformatted) solution reader.

Reads single-block Plot3D ``.q`` solution files written with
``FORM='UNFORMATTED'``.  2-D and 3-D grids are supported.

Binary layout::

    record 1:  nblocks                (1 int32)         -- optional
    record 2:  ni, nj [, nk]          (2 or 3 int32)
    record 3:  mach, alpha, re, time  (4 float64)
    record 4:  dens, xmom, ymom, [zmom,] energy  (all reals, one record)

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
from cfd_io.readers.fortran_binary_sequential import FortranBinaryReader

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# read a Fortran unformatted binary Plot3D solution file
def read_plot3d_flow_binary(
    fpath: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a Fortran unformatted binary Plot3D solution file.

    Args:
        fpath: Path to the binary ``.q`` file.

    Returns:
        Tuple of ``(flow, attrs)`` where:

        - **flow** -- ``{"dens": (ni,nj,nk), "xmom": ..., ...}``
        - **attrs** -- ``{"mach": float, "alpha": float, "re": float, "time": float}``

        For 2-D files *zmom* is omitted and *nk* = 1.
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

        # parse 2-D or 3-D dimensions
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

        # freestream conditions record: mach, alpha, re, time
        freestream = reader.read_reals(expected_count=4)
        attrs = parse_freestream(freestream)
        logger.debug("  freestream: mach=%.4f, alpha=%.2f, re=%.2e, time=%.4e",
                      attrs["mach"], attrs["alpha"], attrs["re"], attrs["time"])

        # flow data -- all conserved variables packed in one Fortran record
        n_vars = 5 if is_3d else 4
        total = ni * nj * nk * n_vars
        flow_data = reader.read_reals(expected_count=total)

    # unpack flat flow vector into separate variable arrays
    flow = unpack_flow(flow_data, ni, nj, nk, is_3d)

    logger.info(
        "read_plot3d_flow_binary: %d vars, shape=(%d,%d,%d) from %s",
        len(flow), ni, nj, nk, fpath,
    )

    return flow, attrs
