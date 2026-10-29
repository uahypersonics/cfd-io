"""HDF5 writer for CFD datasets.

Writes structured grid and flow data into a grouped HDF5 layout with
timestep sub-groups under ``/flow/``.

In-memory cfd-io contract is ``(ni, nj, nk)`` (so callers can write
``arr[i, j, k]`` naturally).  On-disk convention is Fortran-order
``(nk, nj, ni)`` to match Plot3D / Tecplot / CGNS, where the
streamwise index ``i`` is the fastest-varying / contiguous axis.
The writer transposes on write::

    /grid/x              memory (ni, nj, nk) -> disk (nk, nj, ni) float32
    /grid/y              ...
    /flow/00001/uvel     memory (ni, nj, nk) -> disk (nk, nj, ni) float32
    /flow/00001/vvel     ...
    /flow/00002/uvel     ...
    root attrs:          mach, re1, temp_inf, ... (dict values as JSON text)
    /flow/00001 attrs:   iteration, solution_time (if supplied)

Single-timestep data is stored under ``/flow/00001/``.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# axis convention helper: memory -> disk
# --------------------------------------------------
def _mem_to_disk(arr: np.ndarray) -> np.ndarray:
    """Convert an array from in-memory C order to on-disk Fortran order.

    Memory convention (cfd-io ``(ni, nj, nk)``):
        - 1D: ``(ni,)``
        - 2D: ``(ni, nj)``
        - 3D: ``(ni, nj, nk)``

    Disk convention (Plot3D / Tecplot / CGNS):
        - 1D: ``(ni,)``                  (unchanged)
        - 2D: ``(nj, ni)``               (transpose)
        - 3D: ``(nk, nj, ni)``           (reverse axes)

    Returns a contiguous copy so h5py writes a clean dataset.
    """
    # 1d arrays have no axis ambiguity -- return as-is
    if arr.ndim <= 1:
        return np.ascontiguousarray(arr)

    # 2d memory -> disk: simple transpose (ni, nj) -> (nj, ni)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr.T)

    # 3d memory -> disk: reverse axes (ni, nj, nk) -> (nk, nj, ni)
    if arr.ndim == 3:
        return np.ascontiguousarray(arr.transpose(2, 1, 0))

    # higher-dim arrays are not part of the cfd-io contract -- leave alone
    return np.ascontiguousarray(arr)


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

    # encode attributes before opening the file, so a bad value leaves no file behind
    encoded_attrs = _encode_attrs(attrs)

    try:
        # open the HDF5 file and write grid, flow, and attributes
        with h5py.File(fpath, "w") as fobj:

            # -- /grid group (written once) --------------------------------
            grid_grp = fobj.create_group("grid")
            # enforce x, y, z ordering for grid variables
            ordered_keys = [k for k in ("x", "y", "z") if k in grid]
            ordered_keys += [k for k in grid if k not in ordered_keys]
            for name in ordered_keys:
                # transpose from memory (ni, nj, nk) to disk (nk, nj, ni) before write
                grid_grp.create_dataset(name, data=_mem_to_disk(grid[name]), dtype=dtype)

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
                    # transpose from memory (ni, nj, nk) to disk (nk, nj, ni) before write
                    ts_grp.create_dataset(name, data=_mem_to_disk(data), dtype=dtype)

                # per-timestep attributes
                if "_iteration" in ts_dict:
                    ts_grp.attrs["iteration"] = ts_dict["_iteration"]
                if "_solution_time" in ts_dict:
                    ts_grp.attrs["solution_time"] = ts_dict["_solution_time"]

            # -- root attributes -------------------------------------------
            for k, v in encoded_attrs.items():
                # name the attribute when HDF5 can't store its type
                try:
                    fobj.attrs[k] = v
                except TypeError as error:
                    raise TypeError(f"can't write attribute {k!r} to HDF5: {error}") from error
    except Exception:
        # remove the partly written file, then report the original error
        fpath.unlink(missing_ok=True)
        raise

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


# encode attribute values HDF5 can't store natively
def _encode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Prepare root attributes for HDF5: drop ``None``, store dicts as JSON text.

    HDF5 attributes hold scalars, strings and arrays, but no nested
    mappings. Dict values (e.g. ``markers`` with the boundary names) are
    written as JSON text; ``read_hdf5`` turns them back into dicts.

    Args:
        attrs: Root attributes of the dataset.

    Returns:
        Attributes ready to write.

    Raises:
        TypeError: If a dict value can't be encoded as JSON.
    """

    # default: empty
    encoded_attrs: dict[str, Any] = {}

    for key, value in attrs.items():
        # None means "not set": skip
        if value is None:
            continue

        # nested mapping: JSON text
        if isinstance(value, dict):
            try:
                encoded_attrs[key] = json.dumps(value)
            except TypeError as error:
                raise TypeError(f"can't write attribute {key!r} to HDF5: {error}") from error
            continue

        # everything else as is
        encoded_attrs[key] = value

    return encoded_attrs
