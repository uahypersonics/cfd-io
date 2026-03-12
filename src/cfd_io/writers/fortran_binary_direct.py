"""raw binary writer.

Writes structured grid and flow data to the headerless raw binary format and
companion header (text) files that record array dimensions, variable names, info
lines, and timestep indices.

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
# header file writer
# --------------------------------------------------
def _write_header(
    fpath: Path,
    nx: int,
    ny: int,
    nz: int,
    n_params: int,
    info_lines: list[str],
    timesteps: list[int],
    var_names: list[str] | None = None,
) -> None:
    """Write a text header file for a headerless raw binary file.

    Writes the full extended format matching the legacy Fortran output::

         size of array:
         m3 =     1     m2 =   496     m1 =   245
         number of parameters =     6
         number of timesteps  =     1

         Information about file :   (  2  info lines )
         <info line 1>
         Information about parameters :
         pres
         temp
         Numbers of timesteps :
                  1

    Args:
        fpath: Output path for the header file.
        nx: Number of grid points in x (m1).
        ny: Number of grid points in y (m2).
        nz: Number of grid points in z (m3).
        n_params: Number of field variables.
        info_lines: Free-form info lines.
        timesteps: Integer timestep indices.
        var_names: Variable names in ivar order.  Falls back to
            ``var1, var2, ...`` if not provided.
    """

    # write header data
    with open(fpath, "w") as fobj:
        # dimensions
        fobj.write("     size of array:\n")
        fobj.write(f"     m3 = {nz:5d}     m2 = {ny:5d}     m1 = {nx:5d}\n")

        # parameter info
        fobj.write(f"     number of parameters = {n_params:5d}\n")

        # timestep info
        n_timesteps = len(timesteps)
        if n_timesteps < 100000:
            fobj.write(f"     number of timesteps  = {n_timesteps:5d}\n")
        else:
            fobj.write(f"     number of timesteps  = {n_timesteps:6d}\n")

        # info lines section
        fobj.write(" \n")
        fobj.write(f"     Information about file :   ({len(info_lines):3d}  info lines )\n")
        if info_lines:
            for line in info_lines:
                fobj.write(f"   {line}\n")
        else:
            fobj.write(" \n")

        # variable names section
        fobj.write("     Information about parameters :\n")
        if var_names:
            for name in var_names:
                fobj.write(f"   {name}\n")
        else:
            for i in range(1, n_params + 1):
                fobj.write(f"   var{i}\n")

        # timestep indices section
        fobj.write("     Numbers of timesteps :\n")
        # write 6 per line, matching Fortran format (2x,i10)
        for row_start in range(0, len(timesteps), 6):
            chunk = timesteps[row_start : row_start + 6]
            line = "".join(f"  {ts:10d}" for ts in chunk)
            fobj.write(f"{line}\n")

    logger.debug("wrote header: %s", fpath)


# --------------------------------------------------
# binary writer (direct access)
# --------------------------------------------------
def _write_binary(
    fpath: Path,
    arrays: list[np.ndarray],
    dtype: np.dtype,
) -> None:
    """Write 3-D arrays to a flat raw binary file in IOS layout.

    Arrays are written in ivar order.  For each variable the data is
    stored plane-by-plane (XY slices) across all z-indices.  Arrays
    arriving as ``(nx, ny, nz)`` are transposed to the on-disk
    ``(nz, ny, nx)`` order and then flattened.

    Args:
        fpath: Output binary file path.
        arrays: List of 3-D arrays, each ``(nx, ny, nz)``.
        dtype: NumPy dtype for the output (float32 or float64).
    """
    with open(fpath, "wb") as fobj:
        for arr in arrays:
            # convert (nx, ny, nz) -> (nz, ny, nx) for IOS on-disk layout
            reordered = np.transpose(arr, (2, 1, 0))
            # ensure contiguous memory layout and cast to target dtype
            reordered = np.ascontiguousarray(reordered, dtype=dtype)
            # write raw bytes to disk
            reordered.tofile(fobj)

    # debug output for devs
    logger.debug("wrote binary: %s (%d variables, dtype=%s)", fpath, len(arrays), dtype)


# --------------------------------------------------
# public api: binary data writer (direct access)
# --------------------------------------------------
def write_binary_direct(
    fpath: str | Path,
    gpath: str | Path,
    dataset: Dataset,
) -> tuple[Path, Path]:
    """Write a `Dataset` to binary files with corresponding header files.

    Up to four files are created (grid-only or flow-only is allowed):

    - ``<fpath>``: flow binary data (if flow is non-empty)
    - ``<fpath>.cd``: flow header file
    - ``<gpath>``: grid binary data (if grid is non-empty)
    - ``<gpath>.cd``: grid header file

    Args:
        fpath: Path for the flow binary file.
        gpath: Path for the grid binary file.
        dataset: Dataset to write.

    Returns:
        Tuple of ``(fpath, gpath)`` for the two binary files.

    Raises:
        TypeError: If the grid is not a `StructuredGrid`.
        ValueError: If grid and flow are both empty, or array shapes
            are inconsistent.
    """

    # convert to Path objects
    fpath = Path(fpath)
    gpath = Path(gpath)

    if not isinstance(dataset.grid, StructuredGrid):
        raise TypeError("write_binary_direct requires a StructuredGrid")

    # unpack Dataset into dicts
    grid: dict[str, np.ndarray] | None = {
        "x": dataset.grid.x, "y": dataset.grid.y, "z": dataset.grid.z,
    }
    flow: dict[str, np.ndarray] | None = {
        k: v.data for k, v in dataset.flow.items()
    } or None
    attrs = dataset.attrs or {}

    # validate that at least one dict is non-empty
    if not grid and not flow:
        raise ValueError("grid and flow dicts must not both be empty")

    # determine precision from file extension
    if flow:
        fext = fpath.suffix.lower()
    else:
        fext = gpath.suffix.lower()

    if fext == ".s8":
        dtype = np.dtype(np.float64)
    else:
        dtype = np.dtype(np.float32)

    # determine dimensions from the first available array
    if grid:
        dict_tmp = grid
    elif flow:
        dict_tmp = flow

    # grab first array in the dict to check dimensions
    arr_tmp = next(iter(dict_tmp.values()))

    # check that arrays are 3-D
    if arr_tmp.ndim != 3:
        raise ValueError(
            f"expected 3-D grid arrays, got ndim={arr_tmp.ndim}"
        )

    # extract dimensions
    nx, ny, nz = arr_tmp.shape

    # make sure all arrays share the same shape
    dict_tmp: dict[str, np.ndarray] = {}

    # collect all available dicts into dict_tmp for shape checking
    if grid:
        dict_tmp.update(grid)
    if flow:
        dict_tmp.update(flow)

    # check each array against the expected shape
    for var_name, arr in dict_tmp.items():
        if arr.shape != (nx, ny, nz):
            raise ValueError(
                f"shape mismatch for '{var_name}': expected ({nx}, {ny}, {nz}), "
                f"got {arr.shape}"
            )

    # extract metadata (if it exists)

    # get info lines from attrs; default to empty list if not present
    info_lines = attrs.get("info_lines", [])

    # get timesteps from attrs; default to [0] if not present
    timesteps = attrs.get("timesteps", [0])

    # write grid (if present)
    if grid:
        # enforce x, y, z ordering for grid variables
        # start with expected coordinate order: x,y,z
        grid_var_names = []
        for coord in ("x", "y", "z"):
            if coord in grid:
                grid_var_names.append(coord)

        # append any non-coordinate grid variables (e.g. "r", "theta")
        for key in grid:
            if key not in grid_var_names:
                grid_var_names.append(key)

        # build ordered list of arrays matching the variable name order
        grid_arrays = [grid[k] for k in grid_var_names]

        # write headerless binary data
        _write_binary(gpath, grid_arrays, dtype)

        # write header file
        _write_header(
            gpath.with_suffix(".cd"),
            nx=nx, ny=ny, nz=nz,
            n_params=len(grid_arrays),
            var_names=grid_var_names,
            info_lines=info_lines,
            timesteps=[0],
        )

        # debug output for devs
        logger.info(
            "wrote grid: %s  (%d vars, %dx%dx%d, %s)",
            gpath, len(grid), nx, ny, nz, dtype,
        )

    # write flow (if present)
    if flow:
        flow_arrays = list(flow.values())
        flow_var_names = list(flow.keys())

        # write headerless binary data
        _write_binary(fpath, flow_arrays, dtype)

        # write header file
        _write_header(
            fpath.with_suffix(".cd"),
            nx=nx, ny=ny, nz=nz,
            n_params=len(flow_arrays),
            var_names=flow_var_names,
            info_lines=info_lines,
            timesteps=timesteps,
        )

        # debug output for devs
        logger.info(
            "wrote flow: %s  (%d vars, %dx%dx%d, %s)",
            fpath, len(flow), nx, ny, nz, dtype,
        )

    return fpath, gpath
