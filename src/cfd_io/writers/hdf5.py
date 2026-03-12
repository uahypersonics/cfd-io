"""HDF5 writer for CFD datasets.

Writes structured grid and flow data into a grouped HDF5 layout with
timestep sub-groups under ``/flow/``::

    /grid/x              (nx, ny, nz) float32
    /grid/y              (nx, ny, nz) float32
    /flow/00001/uvel     (nx, ny, nz) float32
    /flow/00001/vvel     ...
    /flow/00002/uvel     ...
    root attrs:          mach, re1, temp_inf, ...
    /flow/00001 attrs:   iteration, solution_time (if supplied)

Single-timestep data is stored under ``/flow/00001/``.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# write grid and flow dicts to a grouped HDF5 file
def write_hdf5(
    fpath: str | Path,
    dataset: Dataset,
    *,
    dtype: str = "f",
) -> Path:
    """Write a `Dataset` to a grouped HDF5 file.

    Args:
        fpath: Output file path.
        dataset: Dataset to write.  Flow is stored as a single
            timestep under ``/flow/00001/``.
        dtype: NumPy dtype string for all datasets (default ``"f"`` = float32).

    Returns:
        Path to the created file.
    """
    fpath = Path(fpath)

    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError("write_hdf5 requires a StructuredGrid")

    # unpack Dataset into dicts for writing
    grid = {"x": dataset.grid.x, "y": dataset.grid.y, "z": dataset.grid.z}
    flow: dict[str, np.ndarray] = {k: v.data for k, v in dataset.flow.items()}
    attrs = dataset.attrs or {}

    # normalise flow into {int: dict} form for uniform processing
    timestep_flow = _normalise_flow(flow)

    n_ts = len(timestep_flow)

    # debug output for devs
    logger.debug(
        "write_hdf5: %s  (grid=%d datasets, %d timestep(s))",
        fpath, len(grid), n_ts,
    )

    # open the HDF5 file and write grid, flow, and attributes
    with h5py.File(fpath, "w") as fobj:

        # -- /grid group (written once) --------------------------------
        grid_grp = fobj.create_group("grid")
        # enforce x, y, z ordering for grid variables
        ordered_keys = [k for k in ("x", "y", "z") if k in grid]
        ordered_keys += [k for k in grid if k not in ordered_keys]
        for name in ordered_keys:
            grid_grp.create_dataset(name, data=grid[name], dtype=dtype)

        # -- /flow/<NNNNN>/ groups -------------------------------------
        flow_grp = fobj.create_group("flow")

        # write each timestep as a numbered subgroup (e.g. /flow/00001/)
        for ts_key in sorted(timestep_flow):
            ts_dict = timestep_flow[ts_key]
            ts_name = f"{ts_key:05d}"
            ts_grp = flow_grp.create_group(ts_name)

            # write each flow variable as a dataset in the timestep group
            for name, data in ts_dict.items():
                # skip metadata keys
                if name.startswith("_"):
                    continue
                ts_grp.create_dataset(name, data=data, dtype=dtype)

            # per-timestep attributes
            if "_iteration" in ts_dict:
                ts_grp.attrs["iteration"] = ts_dict["_iteration"]
            if "_solution_time" in ts_dict:
                ts_grp.attrs["solution_time"] = ts_dict["_solution_time"]

        # -- root attributes -------------------------------------------
        if attrs:
            for k, v in attrs.items():
                if v is not None:
                    fobj.attrs[k] = v

    logger.info("wrote %s  (%d timestep(s))", fpath, n_ts)
    return fpath


# --------------------------------------------------
# helpers
# --------------------------------------------------

# convert single-timestep flow dict into multi-timestep form
def _normalise_flow(
    flow: dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    """Convert single-timestep flow dict into multi-timestep form.

    If the first value in *flow* is an ndarray, we assume single-timestep
    and wrap it as ``{1: flow}``.  If it is a dict, we assume the caller
    already provided ``{ts_int: {...}}`` form and return as-is.

    Args:
        flow: Either single-timestep or multi-timestep flow dict.

    Returns:
        Multi-timestep dict ``{int: dict_of_arrays}``.
    """
    if not flow:
        return {1: {}}

    # inspect the first value to determine which form was passed
    first_value = next(iter(flow.values()))

    if isinstance(first_value, np.ndarray):
        # single-timestep dict -- wrap
        return {1: flow}

    # already multi-timestep
    return flow
