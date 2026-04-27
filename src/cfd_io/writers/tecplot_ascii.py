"""Tecplot ASCII (``.dat``) writer.

Writes structured-grid Tecplot ASCII files in POINT data packing.
2-D and 3-D grids are supported.  Grid-only output (no flow variables)
is allowed.

Output format::

    TITLE = "cfd-io"
    VARIABLES = "x", "y", "uvel", "vvel"
    ZONE T="Zone 1", I=100, J=50, K=1, F=POINT
    0.000000E+00  0.000000E+00  1.000000E+00  0.000000E+00
    ...
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# write grid and optional flow to a Tecplot ASCII .dat file
def write_tecplot_ascii(
    fpath: str | Path,
    dataset: Dataset,
    *,
    title: str = "cfd-io",
    zone_title: str = "Zone 1",
) -> Path:
    """Write a `Dataset` to a Tecplot ASCII ``.dat`` file.

    Args:
        fpath: Output file path.
        dataset: Dataset to write.
        title: Tecplot TITLE string.
        zone_title: Tecplot ZONE T string.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If grid is empty or missing ``"x"`` / ``"y"``.
        TypeError: If the grid is not a `StructuredGrid`.
    """
    # convert to Path object
    fpath = Path(fpath)

    # validate grid type
    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError("write_tecplot_ascii requires a StructuredGrid")

    # unpack Dataset into local dicts for writing
    grid = {"x": dataset.grid.x, "y": dataset.grid.y, "z": dataset.grid.z}
    safe_flow = {k: v.data for k, v in dataset.flow.items()}
    safe_attrs = dataset.attrs or {}

    # allow title override from attrs metadata
    if "title" in safe_attrs:
        title = str(safe_attrs["title"])

    # -- determine dimensions from first grid array -----------------------
    # promote 2-D arrays to 3-D with nk=1 so callers may pass either layout;
    # this matches the canonical (ni, nj, nk) convention used elsewhere
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        return arr[..., np.newaxis] if arr.ndim == 2 else arr

    grid = {k: _ensure_3d(v) for k, v in grid.items()}
    safe_flow = {k: _ensure_3d(v) for k, v in safe_flow.items()}

    x = grid["x"]
    if x.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D arrays, got ndim={x.ndim}")

    ni, nj, nk = x.shape

    # -- build ordered variable list: grid coords first, then flow ---------
    var_order: list[str] = []
    all_arrays: list[np.ndarray] = []

    # add grid coordinate variables in canonical order (x, y, z)
    for name in ("x", "y", "z"):
        if name in grid:
            var_order.append(name)
            all_arrays.append(grid[name])

    # add flow variables after grid coordinates
    for name, arr in safe_flow.items():
        var_order.append(name)
        all_arrays.append(arr)

    n_vars = len(var_order)
    n_points = ni * nj * nk

    # -- write file --------------------------------------------------------
    with open(fpath, "w") as fobj:

        # write header: TITLE, VARIABLES, ZONE lines
        fobj.write(f'TITLE = "{title}"\n')

        var_str = ", ".join(f'"{ v}"' for v in var_order)
        fobj.write(f"VARIABLES = {var_str}\n")

        fobj.write(
            f'ZONE T="{zone_title}", I={ni}, J={nj}, K={nk}, F=POINT\n'
        )

        # flatten each array in Fortran order (i-fastest, j, k)
        flat_arrays = []
        for arr in all_arrays:
            flat_arrays.append(arr.ravel(order="F"))

        # write data: one line per grid point, all variables on one line
        for p in range(n_points):
            vals = " ".join(f"{flat_arrays[v][p]:16.8E}" for v in range(n_vars))
            fobj.write(vals + "\n")

    logger.info(
        "wrote %s  (%dx%dx%d, %d vars, POINT format)",
        fpath, ni, nj, nk, n_vars,
    )

    return fpath
