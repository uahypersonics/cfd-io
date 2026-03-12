"""HDF5 reader for CFD datasets.

Reads the grouped timestep layout produced by
``cfd_io.writers.hdf5``::

    /grid/x              (nx, ny, nz)
    /grid/y              (nx, ny, nz)
    /flow/00001/uvel     (nx, ny, nz)
    /flow/00001/vvel     ...
    /flow/00002/uvel     ...
    root attrs:          mach, re1, temp_inf, ...

Also supports two legacy layouts:

1. **Flat flow group** -- ``/flow/uvel`` (no timestep subgroups).
   Treated as a single timestep.
2. **Flat root** -- datasets at ``/``, with ``x``, ``y``, ``z`` sorted
   into grid and everything else into flow.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid
from cfd_io.readers._aliases import GRID_NAMES, normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API: read hdf5 file into dict-of-arrays form
# --------------------------------------------------
def read_hdf5(
    fpath: str | Path,
    timestep: int | None = None,
) -> Dataset:
    """Read a CFD HDF5 file into dict-of-arrays form.

    Automatically detects the layout:

    - **Timestep groups** -- ``/flow/00001/uvel``, ``/flow/00002/uvel``
    - **Flat flow group** -- ``/flow/uvel`` (single timestep)
    - **Flat root** -- ``/uvel`` (single timestep)

    For multi-timestep files the ``flow`` dict uses integer keys::

        {1: {"uvel": ndarray, ...}, 2: {"uvel": ndarray, ...}}

    For single-timestep files (or when *timestep* is specified) the
    ``flow`` dict is flat::

        {"uvel": ndarray, ...}

    Args:
        fpath: Path to the HDF5 file.
        timestep: If given, read only this timestep from a
            multi-timestep file.  Returns flat ``flow`` dict.

    Returns:
        Tuple of ``(grid, flow, attrs)`` where:

        - **grid** -- ``{"x": ndarray, ...}``
        - **flow** -- flat dict (single ts) or ``{int: dict}`` (multi ts)
        - **attrs** -- scalar metadata from root attributes, plus
            ``"timesteps"`` listing available timestep indices when the
            file contains timestep groups

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        KeyError: If *timestep* is given but not present in file.
    """

    # resolve path and check existence
    fpath = Path(fpath)

    if not fpath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {fpath}")

    # debug output for devs
    logger.debug("read_hdf5: %s", fpath)

    # set up empty containers for grid, flow, and attributes
    grid: dict[str, np.ndarray] = {}
    flow: dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]] = {}
    attrs: dict[str, Any] = {}

    # open the HDF5 file and read contents based on detected layout
    with h5py.File(fpath, "r") as fobj:

        # detect layout of hdf5 file

        # check for /grid and /flow groups first (preferred layout)
        has_grid_group = "grid" in fobj and isinstance(fobj["grid"], h5py.Group)
        has_flow_group = "flow" in fobj and isinstance(fobj["flow"], h5py.Group)

        # if grid group exists, read grid datasets into grid dict
        if has_grid_group:
            for name in fobj["grid"]:
                # get grid group
                grid_grp = fobj["grid"]
                # read dataset into numpy array
                grid[name] = np.array(grid_grp[name])
                # debug log for devs
                logger.debug("  grid/%s: shape=%s", name, grid[name].shape)

        # read data
        # three possible layouts:
        # 1. has /flow with timestep subgroups (preferred)
        # 2. has /flow with flat layout (legacy)
        # 3. has no /flow -- treat as empty flow dict
        if has_flow_group:
            # get the flow group
            flow_grp = fobj["flow"]
            # read flow data, handling both timestep-subgroup and flat layouts
            flow = _read_flow_group(flow_grp, timestep)
        elif not has_grid_group:
            # flat root layout (no flow or grid group): may contain both grid and flow datasets at root level
            grid, flow = _read_flat_root(fobj)
        else:
            # has /grid but no /flow -- empty flow
            flow = {}

        # read root attributes into attrs dict
        for key, val in fobj.attrs.items():
            attrs[key] = val
            logger.debug("  attr %s = %s", key, val)

        # add timestep index list when timestep groups are present
        if has_flow_group:

            # get timestep keys from flow group and add to attrs dict for devs (empty if no timestep groups found)
            timestep_keys = _detect_timestep_keys(fobj["flow"])

            if timestep_keys:
                attrs["timesteps"] = timestep_keys

    # summary log
    if isinstance(flow, dict) and flow and isinstance(next(iter(flow.values())), dict):
        n_timesteps = len(flow)
        logger.info(
            "read_hdf5: grid(%d) + flow(%d timesteps) + attrs(%d) from %s",
            len(grid), n_timesteps, len(attrs), fpath,
        )
    else:
        logger.info(
            "read_hdf5: grid(%d) + flow(%d) + attrs(%d) from %s",
            len(grid), len(flow), len(attrs), fpath,
        )

    # warn if no usable data was found -- likely an unsupported layout
    if not grid and not flow:
        logger.warning(
            "read_hdf5: no grid or flow data found in %s -- "
            "file may use an unsupported HDF5 layout",
            fpath,
        )
    elif not flow:
        logger.warning(
            "read_hdf5: grid found but no flow data in %s", fpath,
        )

    # Dataset = single snapshot.  If multi-timestep, pick the first.
    if isinstance(flow, dict) and flow and isinstance(next(iter(flow.values())), dict):
        first_key = min(flow.keys())
        flat_flow: dict[str, np.ndarray] = flow[first_key]
        logger.info("read_hdf5: multi-timestep -- returning timestep %d", first_key)
    else:
        flat_flow = flow

    z = grid.get("z", np.zeros_like(grid["x"])) if grid else np.array([[[0.0]]])
    x = grid.get("x", np.array([[[0.0]]]))
    y = grid.get("y", np.array([[[0.0]]]))

    return Dataset(
        grid=StructuredGrid(x, y, z),
        flow={k: Field(v) for k, v in flat_flow.items()},
        attrs=attrs,
    )


# return sorted list of integer timestep keys from flow group
def _detect_timestep_keys(flow_grp: h5py.Group) -> list[int]:
    """Return sorted list of integer timestep keys found in flow_grp.

    Subgroups whose names are purely numeric (zero-padded or not) are
    treated as timestep groups.  Returns an empty list if none found.
    """

    # initialize empty list for valid timestep keys
    timestep_keys: list[int] = []

    # iterate over items in flow group
    for name in flow_grp:
        # only consider HDF5 groups, not datasets
        if isinstance(flow_grp[name], h5py.Group):
            try:
                # only accept group names that are purely numeric (e.g. "00001", "42") as timestep groups
                # convert to int for sorting
                timestep_keys.append(int(name))
            except ValueError:
                # raise ValueError if group name is not purely numeric, skip it and log for devs (e.g. "metadata" subgroup)
                logger.debug("  skipping non-timestep group '%s' in flow group", name)
                pass

    # sort ascending so timestep order is deterministic
    timestep_keys.sort()

    # return list of sorted timestep keys (empty if none found)
    return timestep_keys


# read /flow group, handling timestep subgroups or flat layout
def _read_flow_group(
    flow_grp: h5py.Group,
    timestep: int | None,
) -> dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]:
    """Read /flow group, handling both timestep-subgroup and flat layouts."""

    # check if flow group contains timestep subgroups (e.g. /flow/00001/)
    timestep_keys = _detect_timestep_keys(flow_grp)

    # no timestep subgroups
    if not timestep_keys:
        # no timestep subgroups found -> flat flow layout (/flow/uvel, /flow/vvel, ...)
        # read all datasets directly from the flow group into a dict
        flow: dict[str, np.ndarray] = {}

        for dataset_name in flow_grp:
            # get dataset
            ds = flow_grp[dataset_name]

            # only read datasets, skip any nested groups
            if isinstance(ds, h5py.Dataset):
                # convert dataset to numpy array
                flow[dataset_name] = np.array(ds)
                logger.debug("  flow/%s: shape=%s", dataset_name, flow[dataset_name].shape)

        # return flow dict
        return flow

    # timestep subgroups found
    if timestep is not None:
        # user requested a specific timestep -> read only requested timestep (if it exits)
        timestep_name = f"{timestep:05d}"

        if timestep_name not in flow_grp:

            # requested timestep not found -> raise KeyError with available timesteps listed for user
            available = ", ".join(str(k) for k in timestep_keys)
            raise KeyError(
                f"timestep {timestep} not found; available: {available}"
            )

        # read specified timestep and return flow dict for that timestep
        flow = _read_single_timestep(flow_grp[timestep_name], timestep)

        return flow

    # only one timestep subgroup -> read it and return flat dict for convenience
    if len(timestep_keys) == 1:
        # read the single available timestep
        timestep_name = f"{timestep_keys[0]:05d}"

        flow = _read_single_timestep(flow_grp[timestep_name], timestep_keys[0])

        return flow

    # multiple timesteps -> return nested dict keyed by timestep index {int: dict}
    flow: dict[int, dict[str, np.ndarray]] = {}

    # loop over all timestep subgroups and read their datasets into the flow dict
    for timestep_key in timestep_keys:

        timestep_name = f"{timestep_key:05d}"

        # read flow group for current timestep
        timestep_grp = flow_grp[timestep_name]

        # read all datasets from this timestep group
        timestep_dict: dict[str, np.ndarray] = {}

        for dataset_name in timestep_grp:
            # get dataset
            ds = timestep_grp[dataset_name]
            # only add verified datasets to timestep_dict, skip any nested groups
            if isinstance(ds, h5py.Dataset):
                timestep_dict[dataset_name] = np.array(ds)

        # add timestep dict to flow dict under the integer timestep key
        flow[timestep_key] = timestep_dict

        # debug log for devs
        logger.debug("  flow/%s: %d variables", timestep_name, len(timestep_dict))

    # return nested flow dict keyed by timestep index
    return flow


# read all datasets from one timestep group into a flat dict
def _read_single_timestep(
    timestep_grp: h5py.Group,
    timestep_key: int,
) -> dict[str, np.ndarray]:
    """Read all datasets from a single timestep group into a flat dict."""

    # initialize empty dict for this timestep
    result: dict[str, np.ndarray] = {}

    # read all datasets from this timestep group into the result dict
    for dataset_name in timestep_grp:
        # get dataset
        ds = timestep_grp[dataset_name]
        # only add verified datasets to result, skip any nested groups
        if isinstance(ds, h5py.Dataset):
            result[dataset_name] = np.array(ds)
            # debug log for devs
            logger.debug("  flow/%05d/%s: shape=%s", timestep_key, dataset_name, result[dataset_name].shape)

    # return the dict of flow variables for this timestep
    return result


# read flat-layout file where all datasets sit at root level
def _read_flat_root(
    fobj: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read flat-layout file where all datasets sit at root level.

    Returns:
        Tuple of ``(grid, flow)`` dicts.
    """

    # generate empty grid and flow dicts (fill where appropriate based on variable names, only one may exist)
    grid: dict[str, np.ndarray] = {}
    flow: dict[str, np.ndarray] = {}

    # iterate over datseets at root level
    for dataset_name in fobj:

        # get dataset
        ds = fobj[dataset_name]

        # skip non-dataset items at root level
        if not isinstance(ds, h5py.Dataset):
            continue

        # read dataset into numpy array
        arr = np.array(ds)

        # normalize the variable name via the central alias table
        var_name_normalized = normalize(dataset_name).lower()

        if var_name_normalized in GRID_NAMES:
            # add recognized grid variable to grid dict
            grid[var_name_normalized] = arr
            logger.debug("  grid/%s (flat): shape=%s", var_name_normalized, arr.shape)

        else:
            # add all other variables to flow dict
            flow[var_name_normalized] = arr

            # log variable name changes for devs
            if var_name_normalized != dataset_name:
                logger.debug("  flow/%s -> %s (flat): shape=%s", dataset_name, var_name_normalized, arr.shape)
            else:
                logger.debug("  flow/%s (flat): shape=%s", var_name_normalized, arr.shape)

    # return grid and flow dicts
    return grid, flow
