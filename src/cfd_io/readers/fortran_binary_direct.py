"""raw binary reader.

the raw binary is a headerless raw binary dump.
A companion text file provides the grid dimensions, variable
names, info lines, and timestep indices.

On-disk layout (outer -> inner): **time, parameter, z, y, x**.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.readers._aliases import normalize

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# BinaryHeader dataclass
# --------------------------------------------------
@dataclass
class BinaryHeader:
    """Grid dimensions and metadata parsed from a header file.

    Attributes:
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        nz: Number of grid points in z.
        nt: Number of timesteps.
        np: Number of field variables (parameters).
        var_names: Variable names in ivar order (from the header file).
        info_lines: Free-form info lines from the header file.
        timesteps: Integer timestep indices from the header file.
        precision: ``"float64"`` or ``"float32"``.
    """

    # define attributes with default values; will be populated by read_header
    nx: int = 0
    ny: int = 0
    nz: int = 0
    nt: int = 0
    np: int = 0
    var_names: list[str] = field(default_factory=list)
    info_lines: list[str] = field(default_factory=list)
    timesteps: list[int] = field(default_factory=list)
    precision: str = "float64"
    is_byteswapped: bool | None = None

    # dtype property to get the NumPy dtype corresponding to the precision string
    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype corresponding to the file precision."""

        if self.precision == "float32":
            return np.dtype(np.float32)
        return np.dtype(np.float64)

    # bytes per value is derived from the dtype itemsize property
    @property
    def bytes_per_value(self) -> int:
        """Bytes consumed by one scalar value on disk."""
        return self.dtype.itemsize

    # repr method for debugging purposes
    def __repr__(self) -> str:
        return (
            f"BinaryHeader(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"nt={self.nt}, np={self.np}, precision={self.precision!r})"
        )

    # byteswap: check once, cache result, apply on all subsequent reads
    def apply_byteswap(self, vals: np.ndarray) -> np.ndarray:
        """Byteswap *vals* if needed, caching the result for future calls."""
        if self.is_byteswapped is None:
            self.is_byteswapped = _check_byteswap(vals)

        if self.is_byteswapped:
            vals = vals.byteswap()

        return vals


# --------------------------------------------------
# read header file
# --------------------------------------------------
def read_header(fpath: str | Path) -> BinaryHeader:
    """Read grid dimensions and metadata from a header file.

    Parses both the minimal format (4 lines) and the extended format
    that includes variable names, info lines, and timestep indices.

    Expected extended format::

         size of array:
         m3 =     1     m2 =   496     m1 =   245
         number of parameters =     6
         number of timesteps  =     1

         Information about file :   (  2  info lines )
         <info line 1>
         <info line 2>
         Information about parameters :
         pres
         temp
         uvel
         Numbers of timesteps :
                  1

    Args:
        fpath: Path to the header file.

    Returns:
        Parsed header with grid dimensions and metadata.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
    """

    # convert to Path object
    fpath = Path(fpath)

    # raise error if file does not exist
    if not fpath.exists():
        raise FileNotFoundError(f"Header file not found: {fpath}")

    # debug output for devs
    logger.debug("reading header from: %s", fpath)

    # instantiate a BinaryHeader dataclass to populate and return
    header = BinaryHeader()

    # infer precision from companion binary file extension

    # get stem (file name without extension)
    stem = fpath.stem

    if fpath.parent.joinpath(f"{stem}.s4").exists():
        # .s4 companion -> single precision
        header.precision = "float32"
    elif fpath.parent.joinpath(f"{stem}.s8").exists():
        # .s8 companion -> double precision
        header.precision = "float64"
    else:
        raise FileNotFoundError(
            f"no companion binary file found for header: {fpath}"
        )

    # read all lines from the header file
    with open(fpath) as fobj:
        lines = fobj.readlines()

    # parse lines and extract dimensions, variable names, info lines, and timesteps
    # line 0: "     size of array:"
    # line 1: "     m3 =     1     m2 =   496     m1 =   245"

    # get dim_line and strip leading/trailing whitespace
    dim_line = lines[1].strip()

    # debug output for devs
    logger.debug("dimensions line: %s", dim_line)

    # split the dimensions line into parts
    parts = dim_line.split()

    # get index for m3 value (m3 string followed by "=" followed by value => +2)
    m3_idx = parts.index("m3") + 2
    # get m3 value and convert to int
    m3 = int(parts[m3_idx])
    # get index for m2 value (m2 string followed by "=" followed by value => +2)
    m2_idx = parts.index("m2") + 2
    # get m2 value and convert to int
    m2 = int(parts[m2_idx])
    # get index for m1 value (m1 string followed by "=" followed by value => +2)
    m1_idx = parts.index("m1") + 2
    # get m1 value and convert to int
    m1 = int(parts[m1_idx])

    # save dimensions to header
    header.nx = m1
    header.ny = m2
    header.nz = m3

    # get params line and strip leading/trailing whitespace

    # line 2: e.g. "     number of parameters =     6"
    params_line = lines[2].strip()

    # split into parts
    parts = params_line.split("=")

    # save number of parameters to header
    header.np = int(parts[1].strip())

    # get timesteps line and strip leading/trailing whitespace

    # line 3: e.g. "     number of timesteps  =     1"
    timestep_line = lines[3].strip()

    # split into parts
    parts = timestep_line.split("=")

    # save number of timesteps to header
    header.nt = int(parts[1].strip())

    # find information about file (info lines)
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("information about file"):
            # extract count from parenthesised portion, e.g. "(  2  info lines )"
            m = re.search(r"\(\s*(\d+)\s*info lines?\s*\)", line, re.IGNORECASE)
            if m:
                n_info = int(m.group(1))
                for i in range(n_info):
                    header.info_lines.append(lines[idx + 1 + i].strip())
            break

    # find information about parameters
    for idx, line in enumerate(lines):

        if line.strip().lower().startswith("information about parameters"):

            # next np lines are variable names
            for i in range(header.np):
                # get var_line and strip leading/trailing whitespace
                var_line = lines[idx + 1 + i].strip()
                # normalize the variable name via the central alias table and save to header (see _aliases.py)
                var_name = normalize(var_line)
                # append var_name to header.var_names
                header.var_names.append(var_name)
            break

    # find information about timesteps (skip the 4 required header lines so we
    # don't accidentally match the "number of timesteps = N" line at index 3)
    for idx, line in enumerate(lines):
        if idx < 4:
            continue

        if line.strip().lower().startswith("numbers of timesteps") or line.strip().lower().startswith("number of timesteps"):

            # remaining lines are timestep indices, 6 per line
            for j in range(idx + 1, len(lines)):
                timestep_line = lines[j].strip()
                if not timestep_line:
                    continue
                # split line into parts and convert to integers
                parts = timestep_line.split()
                for p in parts:
                    header.timesteps.append(int(p.strip()))
            break

    # debug output for devs
    logger.debug("parsed header: %s", header)
    if header.var_names:
        logger.debug("  var_names: %s", header.var_names)
    if header.timesteps:
        logger.debug("  timesteps: %d entries", len(header.timesteps))

    # return the populated header dataclass
    return header


# --------------------------------------------------
# byteswap detection
# --------------------------------------------------
def _check_byteswap(
    vals: np.ndarray,
    threshold: int = 35,
) -> bool:
    """Check if raw binary data needs byte-swapping.

    Uses a two-pass strategy:

    1. Sample non-zero, finite values from the array.
    2. If any exponent exceeds *threshold*, perform a secondary check
       by actually byte-swapping and comparing whether the exponent
       improves (gets smaller).  This avoids false positives from
       legitimately large values.
    3. All-NaN / all-Inf data is treated as needing a swap.

    Args:
        vals: 1-D array of raw values read from disk.
        threshold: Exponent magnitude above which a value is suspicious.
            The legacy Fortran code uses 35.

    Returns:
        ``True`` if the data appears byte-swapped.
    """

    # initialize is_swapped flag to False; will set to True if any checks indicate swapping is needed
    is_swapped = False

    # find non-zero, finite values to sample
    test_vals = np.isfinite(vals) & (vals != 0)

    # all zero or NaN/Inf
    if not np.any(test_vals):
        # if non finite values are present we likely need to byteswap
        is_inf = np.any(~np.isfinite(vals))

        # debug output for devs
        if is_inf:
            logger.debug("byteswap check: all non-finite values => needs swap")
            is_swapped = True

        return is_swapped

    # sample up to 64 values for efficiency (if we check too many values it gets slow)
    sample = vals[test_vals][:64]

    # compute exponents of the sample values
    abs_sample = np.abs(sample)
    log_vals = np.log10(abs_sample)
    exponents = np.abs(np.floor(log_vals))

    # primary check: if any exponent exceeds the threshold, the data may be swapped
    large_exponents = exponents > threshold

    # no large exponents detected (passed primary check) -> data is likely not swapped
    if not np.any(large_exponents):
        return is_swapped

    # secondary check: large exponents detected -> try byte-swapping and see if exponent improves

    # grab only values with large exponents
    test_vals = sample[large_exponents]

    # byteswap the test values for the secondary check
    test_vals_swapped = test_vals.byteswap()

    # only consider finite, non-zero swapped values for the exponent comparison
    test_vals_swapped_filtered = np.isfinite(test_vals_swapped) & (test_vals_swapped != 0)

    # swapped values are all zero or NaN/Inf -> original was definitely swapped data
    if not np.any(test_vals_swapped_filtered):
        # debug output for devs
        logger.debug("byteswap check: swapped values are NaN => needs swap")

        is_swapped = True

        # return is_swapped flags (at this point it likely is swapped)
        return is_swapped

    # none of the above detections have triggered yet -> compare exponents before vs after swapping

    # compute exponents of the swapped test values
    swapped_subset = test_vals_swapped[test_vals_swapped_filtered]
    abs_swapped = np.abs(swapped_subset)
    log_swapped = np.log10(abs_swapped)
    exponents_swapped = np.abs(np.floor(log_swapped))

    # get exponents of original values corresponding to the swapped values that passed the finite, non-zero filter
    exponents_original = exponents[large_exponents][test_vals_swapped_filtered]

    # if swapping reduces the average exponent, data likely needs swapping
    is_swapped = bool(np.mean(exponents_swapped) < np.mean(exponents_original))

    # debug output for devs
    if is_swapped:
        logger.debug(
            "byteswap check: avg exponent %.1f -> %.1f after swap => needs swap",
            np.mean(exponents_original), np.mean(exponents_swapped),
        )

    # return is_swapped flag
    return is_swapped


# --------------------------------------------------
# function to compute byte offset for a given (it, ivar, k, j) index
# required for direct access of data
# --------------------------------------------------
def _byte_offset(
    header: BinaryHeader,
    it: int,
    ivar: int,
    k: int = 1,
    j: int = 1,
) -> int:
    """Compute the byte offset for a direct access binary file.

    On-disk layout (outer -> inner): time, parameter, z, y, x.
    All indices are 1-based.

    Args:
        header: Parsed header file.
        it: Timestep index (1-based).
        ivar: Variable index (1-based).
        k: Z-index (1-based).  Defaults to 1 (start of variable block).
        j: Y-index (1-based).  Defaults to 1 (start of k-plane).

    Returns:
        Byte offset from the start of the file.
    """

    # get dimensions and store in local variables for readability
    nx = header.nx
    ny = header.ny
    nz = header.nz
    n_params = header.np

    # convenience alias to use downstream
    bpv = header.bytes_per_value

    # element offset (number of scalar values before the target)
    elem_offset = (
        (it - 1) * n_params * nx * ny * nz
        + (ivar - 1) * nx * ny * nz
        + (k - 1) * nx * ny
        + (j - 1) * nx
    )

    return elem_offset * bpv


# --------------------------------------------------
# read a single x-line from a binary file
# --------------------------------------------------
def read_binary_direct_x_line(
    fpath: str | Path,
    header: BinaryHeader,
    j: int,
    k: int,
    ivar: int,
    it: int,
) -> np.ndarray:
    """Read a single x-line from an IOS binary file.

    Args:
        fpath: Path to the binary data file.
        header: Parsed header from ``read_header``.
        j: Y-index (1-based).
        k: Z-index (1-based).
        ivar: Variable index (1-based).
        it: Timestep index (1-based).

    Returns:
        1-D array with shape ``(nx,)``.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        ValueError: If any index is out of range.
    """

    # convert to Path object
    fpath = Path(fpath)

    # check if path is valid
    if not fpath.exists():
        raise FileNotFoundError(f"file not found: {fpath}")

    # guard against out of range indices
    if j > header.ny:
        raise ValueError(f"j index {j} out of range (ny={header.ny})")
    if k > header.nz:
        raise ValueError(f"k index {k} out of range (nz={header.nz})")
    if ivar > header.np:
        raise ValueError(f"variable index {ivar} out of range (np={header.np})")
    if it > header.nt:
        raise ValueError(f"time index {it} out of range (nt={header.nt})")

    # compute offset for reading the right chunk of data for the given indices
    offset = _byte_offset(header, it=it, ivar=ivar, k=k, j=j)

    # read binary data directly into a 1-D array of length nx
    with open(fpath, "rb") as fobj:
        fobj.seek(offset)
        vals = np.fromfile(fobj, dtype=header.dtype, count=header.nx)

    # check if byte-swapping is needed and apply if so
    vals = header.apply_byteswap(vals)

    return vals


# --------------------------------------------------
# read an xy plane from a binary file
# --------------------------------------------------
def read_binary_direct_xy_plane(
    fpath: str | Path,
    header: BinaryHeader,
    k: int,
    ivar: int,
    it: int,
) -> np.ndarray:
    """Read an xy plane from a binary file.

    Args:
        fpath: Path to the binary data file.
        header: Parsed header from ``read_header``.
        k: Z-index (1-based).
        ivar: Variable index (1-based).
        it: Timestep index (1-based).

    Returns:
        2-D array with shape ``(ny, nx)``.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        ValueError: If any index is out of range.
    """

    # convert to Path object
    fpath = Path(fpath)

    # check if path is valid
    if not fpath.exists():
        raise FileNotFoundError(f"file not found: {fpath}")

    # guard against out of range indices
    if k > header.nz:
        raise ValueError(f"k index {k} out of range (nz={header.nz})")
    if ivar > header.np:
        raise ValueError(f"variable index {ivar} out of range (np={header.np})")
    if it > header.nt:
        raise ValueError(f"time index {it} out of range (nt={header.nt})")

    # compute offset for reading the right chunk of data for the given indices
    offset = _byte_offset(header, it=it, ivar=ivar, k=k)

    # read binary data directly into a 1-D array of length nx*ny
    with open(fpath, "rb") as fobj:
        fobj.seek(offset)
        vals = np.fromfile(fobj, dtype=header.dtype, count=header.nx * header.ny)

    # byteswap if needed
    vals = header.apply_byteswap(vals)

    # return array reshaped to (ny, nx)
    return vals.reshape((header.ny, header.nx))


# --------------------------------------------------
# read a full 3d volume for one variable from a binary file
# --------------------------------------------------
def read_binary_direct_xyz_volume(
    fpath: str | Path,
    header: BinaryHeader,
    ivar: int,
    it: int,
) -> np.ndarray:
    """Read a full 3-D volume for one variable from a binary file.

    Args:
        fpath: Path to the binary data file.
        header: Parsed header from ``read_header``.
        ivar: Variable index (1-based).
        it: Timestep index (1-based).

    Returns:
        3-D array with shape ``(nx, ny, nz)``.

    Raises:
        FileNotFoundError: If *fpath* does not exist.
        ValueError: If any index is out of range.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"file not found: {fpath}")

    if ivar > header.np:
        raise ValueError(f"variable index {ivar} out of range (np={header.np})")
    if it > header.nt:
        raise ValueError(f"time index {it} out of range (nt={header.nt})")

    offset = _byte_offset(header, it=it, ivar=ivar)

    with open(fpath, "rb") as fobj:
        fobj.seek(offset)
        vals = np.fromfile(fobj, dtype=header.dtype, count=header.nx * header.ny * header.nz)

    vals = header.apply_byteswap(vals)

    # on-disk order is (nz, ny, nx) -- reshape then transpose to (nx, ny, nz)
    volume = vals.reshape((header.nz, header.ny, header.nx))
    volume = np.transpose(volume, (2, 1, 0))

    return volume


# --------------------------------------------------
# read bindary data with direct access, returning dict-of-arrays for grid and flow variables
# --------------------------------------------------
def read_binary_direct(
    fpath: str | Path,
    gpath: str | Path,
    *,
    it: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read IOS grid and flow binary files into dict-of-arrays form.

    Reads the companion ``.cd`` header files to determine grid
    dimensions and variable names.

    Args:
        fpath: Path to the flow binary file.
        gpath: Path to the grid binary file.
        it: Timestep index (1-based).  Default 1.

    Returns:
        Tuple of ``(grid, flow, attrs)`` where:

        - **grid** -- ``{"x": ndarray, "y": ndarray}`` each ``(nx, ny, nz)``
        - **flow** -- ``{"pres": ndarray, ...}`` each ``(nx, ny, nz)``
        - **attrs** -- dict with ``"info_lines"``,
            ``"timesteps"`` if present in the ``.cd`` file

    Raises:
        FileNotFoundError: If any input file or its ``.cd`` companion
            is missing.
        ValueError: If a ``.cd`` header does not contain variable names.
    """

    # ensure input paths are Path objects for consistent handling
    fpath = Path(fpath)
    gpath = Path(gpath)

    # read grid header file
    gpath_header = gpath.with_suffix(".cd")
    grid_header = read_header(gpath_header)

    # debug output for devs
    logger.info("reading grid from %s  (%s)", gpath, grid_header)

    # variable names must be present in the header file otherwise data cannot be read
    if not grid_header.var_names:
        raise ValueError(
            f"grid .cd header has no variable names: {gpath_header}"
        )

    # conviencience alias for grid variable names (e.g. "x", "y", "z")
    grid_names = grid_header.var_names

    # intiailze empty grid dict
    grid: dict[str, np.ndarray] = {}

    # read each grid variable into the grid dict using the variable names from the header
    for ivar, var_name in enumerate(grid_names, 1):
        # read data
        grid[var_name] = read_binary_direct_xyz_volume(gpath, grid_header, ivar=ivar, it=1)
        logger.info("  grid/%s: shape=%s", var_name, grid[var_name].shape)

    # read flow header file
    fpath_header = fpath.with_suffix(".cd")
    flow_header = read_header(fpath_header)

    # debug output for devs
    logger.info("reading flow from %s  (%s)", fpath, flow_header)

    # variable names must be present in the header file otherwise data cannot be read
    if not flow_header.var_names:
        raise ValueError(
            f"flow .cd header has no variable names: {fpath_header}"
        )

    # conviencience alias for flow variable names (e.g. "pres", "vel")
    flow_names = flow_header.var_names

    # intiailze empty flow dict
    flow: dict[str, np.ndarray] = {}

    for ivar, var_name in enumerate(flow_names, 1):
        # read data
        flow[var_name] = read_binary_direct_xyz_volume(fpath, flow_header, ivar=ivar, it=it)
        logger.info("  flow/%s: shape=%s", var_name, flow[var_name].shape)

    # build attrs dict from metadata
    attrs: dict[str, Any] = {}
    if flow_header.info_lines:
        attrs["info_lines"] = flow_header.info_lines
    if flow_header.timesteps:
        attrs["timesteps"] = flow_header.timesteps

    # debug output for devs
    logger.info("loaded grid (%d vars) + flow (%d vars)", len(grid), len(flow))

    return grid, flow, attrs
