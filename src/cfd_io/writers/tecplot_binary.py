"""Tecplot binary (``.plt``) writer via pytecplot.

Writes structured-grid Tecplot ``.plt`` files using the ``pytecplot``
library (requires a Tecplot 360 license).

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

from cfd_io.dataset import Dataset, StructuredGrid

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

# write a Tecplot binary .plt file via pytecplot
def write_tecplot_plt(
    fpath: str | Path,
    dataset: Dataset,
) -> Path:
    """Write a `Dataset` to a Tecplot binary ``.plt`` file.

    Requires ``pytecplot`` (optional dependency).  Creates a single
    ordered zone with all grid and flow variables.

    Args:
        fpath: Output ``.plt`` file path.
        dataset: Dataset to write.

    Raises:
        ImportError: If ``pytecplot`` is not installed.
        TypeError: If the grid is not a `StructuredGrid`.
    """
    # check that pytecplot is available before proceeding
    _require_pytecplot()

    import tecplot as tp
    from tecplot.constant import FieldDataType

    # convert to Path object
    fpath = Path(fpath)

    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError("write_tecplot_plt requires a StructuredGrid")

    # unpack Dataset
    grid = {"x": dataset.grid.x, "y": dataset.grid.y, "z": dataset.grid.z}
    flow = {k: v.data for k, v in dataset.flow.items()}
    attrs = dataset.attrs or {}

    # build ordered variable name list: grid vars first (x, y, z), then flow
    grid_names = [k for k in ("x", "y", "z") if k in grid]
    grid_names += [k for k in grid if k not in grid_names]
    flow_names = list(flow.keys())
    all_names = grid_names + flow_names

    # determine zone shape from first grid variable
    first_arr = next(iter(grid.values()))
    shape = first_arr.shape

    # ensure 3-D shape tuple for add_ordered_zone (2-D grids get nk=1)
    if len(shape) == 2:
        ni, nj = shape
        zone_shape = (ni, nj, 1)
    elif len(shape) == 3:
        zone_shape = shape
    else:
        raise ValueError(f"unexpected grid array ndim={len(shape)}")

    zone_title = attrs.get("zone_title", "Zone 1")

    # debug output for devs
    logger.debug(
        "write_tecplot_plt: %s, shape=%s, vars=%d",
        fpath, zone_shape, len(all_names),
    )

    # create dataset and ordered zone via pytecplot
    frame = tp.active_frame()
    dataset = frame.create_dataset("CFD Data", all_names)

    dtypes = [FieldDataType.Double] * len(all_names)
    zone = dataset.add_ordered_zone(zone_title, zone_shape, dtypes=dtypes)

    # populate grid variables (ravel in Fortran order for pytecplot)
    for name in grid_names:
        arr = grid[name]
        zone.values(name)[:] = arr.ravel(order="F")

    # populate flow variables
    for name in flow_names:
        arr = flow[name]
        zone.values(name)[:] = arr.ravel(order="F")

    # write the .plt file to disk
    tp.data.save_tecplot_plt(str(fpath), dataset=dataset)

    logger.info(
        "write_tecplot_plt: wrote %s (grid=%d, flow=%d)",
        fpath, len(grid), len(flow),
    )

    return fpath
