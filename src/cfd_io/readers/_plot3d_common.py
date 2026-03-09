"""Shared helpers for Plot3D readers."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------
# coordinate unpacking
# --------------------------------------------------

# split a flat coordinate vector into separate x, y, z arrays
def unpack_coordinates(
    data: np.ndarray,
    ni: int,
    nj: int,
    nk: int,
    is_3d: bool,
) -> dict[str, np.ndarray]:
    """Split a flat coordinate array into x, y [, z] dict entries.

    Plot3D stores coordinates in Fortran column-major order:
    all x values first (i-fastest), then all y, then all z.

    Args:
        data: 1-D array of all coordinate values concatenated.
        ni: Number of points in i-direction.
        nj: Number of points in j-direction.
        nk: Number of points in k-direction.
        is_3d: If True, expect z coordinates in *data*.

    Returns:
        Grid dict ``{"x": (ni, nj, nk), "y": ..., "z": ...}``.
    """

    # each coordinate block is ni*nj*nk values in Fortran column-major order
    block_size = ni * nj * nk
    offset = 0

    # extract x coordinate block and reshape to 3-D array
    x = data[offset : offset + block_size].reshape((ni, nj, nk), order="F")
    offset += block_size

    # extract y coordinate block and reshape to 3-D array
    y = data[offset : offset + block_size].reshape((ni, nj, nk), order="F")
    offset += block_size

    grid: dict[str, np.ndarray] = {"x": x, "y": y}

    # extract z coordinate block if 3-D
    if is_3d:
        z = data[offset : offset + block_size].reshape((ni, nj, nk), order="F")
        grid["z"] = z

    return grid


# --------------------------------------------------
# flow variable unpacking
# --------------------------------------------------

# standard Plot3D .q variable names in storage order
_Q_VARS_3D: list[str] = ["dens", "xmom", "ymom", "zmom", "energy"]
_Q_VARS_2D: list[str] = ["dens", "xmom", "ymom", "energy"]


# split a flat flow vector into separate variable arrays
def unpack_flow(
    data: np.ndarray,
    ni: int,
    nj: int,
    nk: int,
    is_3d: bool,
) -> dict[str, np.ndarray]:
    """Split a flat flow array into named variable dict entries.

    Plot3D .q files store conserved variables in Fortran column-major
    order: all density values first, then x-momentum, y-momentum,
    [z-momentum], then total energy.

    Args:
        data: 1-D array of all flow values concatenated.
        ni: Number of points in i-direction.
        nj: Number of points in j-direction.
        nk: Number of points in k-direction.
        is_3d: If True, expect 5 variables (with z-momentum);
               otherwise 4 variables.

    Returns:
        Flow dict, e.g. ``{"dens": (ni,nj,nk), "xmom": ..., ...}``.
    """
    block_size = ni * nj * nk
    var_names = _Q_VARS_3D if is_3d else _Q_VARS_2D

    flow: dict[str, np.ndarray] = {}
    offset = 0
    for name in var_names:
        flow[name] = data[offset : offset + block_size].reshape(
            (ni, nj, nk), order="F",
        )
        offset += block_size

    return flow


# parse the 4-value freestream conditions record
def parse_freestream(values: np.ndarray) -> dict[str, Any]:
    """Parse the Plot3D freestream conditions into an attrs dict.

    Standard .q header contains 4 floats: mach, alpha, re, time.

    Args:
        values: 1-D array of 4 floats.

    Returns:
        Dict with keys ``mach``, ``alpha``, ``re``, ``time``.
    """
    return {
        "mach": float(values[0]),
        "alpha": float(values[1]),
        "re": float(values[2]),
        "time": float(values[3]),
    }
