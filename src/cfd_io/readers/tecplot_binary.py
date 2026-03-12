"""Tecplot binary (``.plt``) reader via pytecplot.

Reads structured-grid Tecplot ``.plt`` files using the ``pytecplot``
library (requires a Tecplot 360 license).  Only the first zone is read.

Install the optional dependency::

    pip install cfd-io[tecplot]

Or directly::

    pip install pytecplot
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers._aliases import GRID_NAMES, normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# dependency check
# --------------------------------------------------

# raise a clear error if pytecplot is not installed
def _require_pytecplot() -> None:
    """Raise a clear error if pytecplot is not installed."""
    try:
        import tecplot  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pytecplot is required for .plt support. "
            "Install it with: pip install cfd-io[tecplot]"
        ) from exc


# --------------------------------------------------
# public API
# --------------------------------------------------

# read a Tecplot binary .plt file via pytecplot
def read_tecplot_plt(
    fpath: str | Path,
) -> Dataset:
    """Read a Tecplot binary ``.plt`` file.

    Requires ``pytecplot`` (optional dependency).  Only the first zone
    is read.  Variables named ``x``, ``y``, ``z`` (case-insensitive)
    are placed in the grid dict; everything else goes to flow.

    Args:
        fpath: Path to the ``.plt`` file.

    Returns:
        Tuple of ``(grid, flow, attrs)`` where:

        - **grid** -- ``{"x": (ni, nj, nk), ...}``
        - **flow** -- ``{"uvel": (ni, nj, nk), ...}``
        - **attrs** -- ``{"zone_title": str}`` if present

    Raises:
        ImportError: If ``pytecplot`` is not installed.
        FileNotFoundError: If *fpath* does not exist.
    """
    _require_pytecplot()

    import tecplot as tp
    from tecplot.constant import ReadDataOption

    fpath = Path(fpath)

    if not fpath.exists():
        raise FileNotFoundError(f"Tecplot PLT file not found: {fpath}")

    logger.debug("read_tecplot_plt: %s", fpath)

    # load file into pytecplot -- replace any existing data
    dataset = tp.data.load_tecplot(
        str(fpath),
        read_data_option=ReadDataOption.ReplaceInActiveFrame,
    )

    if dataset.num_zones == 0:
        raise ValueError(f"no zones found in {fpath}")

    # read first zone only
    zone = dataset.zone(0)
    zone_name = zone.name

    # determine dimensions
    ni = zone.dimensions[0]
    nj = zone.dimensions[1]
    nk = zone.dimensions[2] if len(zone.dimensions) > 2 else 1

    logger.debug("  zone '%s': %dx%dx%d, %d vars", zone_name, ni, nj, nk, dataset.num_variables)

    # extract variable data
    grid: dict[str, np.ndarray] = {}
    flow: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}

    if zone_name:
        attrs["zone_title"] = zone_name

    for var in dataset.variables():
        var_name = var.name
        arr_flat = zone.values(var).as_numpy_array()

        # reshape to (ni, nj, nk) in Fortran order (i-fastest)
        arr = arr_flat.reshape((ni, nj, nk), order="F")

        canonical = normalize(var_name)
        if canonical.lower() in GRID_NAMES:
            grid[canonical.lower()] = arr
        else:
            flow[canonical] = arr

        logger.debug("  %s: shape=%s", var_name, arr.shape)

    logger.info(
        "read_tecplot_plt: grid(%d) + flow(%d) from %s",
        len(grid), len(flow), fpath,
    )

    return Dataset(
        grid=StructuredGrid(grid["x"], grid["y"], grid.get("z", np.zeros_like(grid["x"]))),
        flow={k: Field(v) for k, v in flow.items()},
        attrs=attrs,
    )
