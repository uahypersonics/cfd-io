"""Extension-based dispatchers for reading and writing CFD data.

read_file and write_file pick the reader or writer by the file extension, from the
tables _READERS and _WRITERS below. Supported formats:

========================  ======================  ====  =====  =====================================
Extension                 Format                  Read  Write  Notes
========================  ======================  ====  =====  =====================================
``.s8``                   raw binary, float64     yes   yes    split: flow file + grid file
``.s4``                   raw binary, float32     yes   yes    split: flow file + grid file
``.h5``, ``.hdf5``        HDF5                    yes   yes    write precision: ``dtype``
``.x``, ``.xyz``          Plot3D grid             yes   yes    grid only; written binary
``.p2dfmt``, ``.p3dfmt``  Plot3D grid             yes   yes    grid only; written formatted (ASCII)
``.q``                    Plot3D solution         yes   no     flow only
``.dat``                  Tecplot ASCII           yes   yes
``.plt``                  Tecplot binary          yes   yes    requires pytecplot
``.vtu``                  VTK unstructured grid   yes   no     structured quads rebuilt (SU2 output)
``.cgns``                 CGNS                    yes   no
``.su2``                  SU2 mesh                yes   yes    read: 2D quads; write: 2D and 3D
``.msh``                  Gmsh mesh               no    yes    2D and 3D; order 2: call write_gmsh
========================  ======================  ====  =====  =====================================
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cfd_io.dataset import Dataset, StructuredGrid
from cfd_io.readers.cgns import read_cgns
from cfd_io.readers.fortran_binary_direct import read_binary_direct
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.readers.plot3d import read_plot3d
from cfd_io.readers.plot3d_flow import read_plot3d_flow
from cfd_io.readers.su2 import read_su2
from cfd_io.readers.tecplot_ascii import read_tecplot_ascii
from cfd_io.readers.tecplot_binary import read_tecplot_plt
from cfd_io.readers.vtu import read_vtu
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.gmsh import write_gmsh
from cfd_io.writers.hdf5 import write_hdf5
from cfd_io.writers.plot3d import write_plot3d
from cfd_io.writers.su2 import write_su2
from cfd_io.writers.tecplot_ascii import write_tecplot_ascii
from cfd_io.writers.tecplot_binary import write_tecplot_plt

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# format registry
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class _FormatEntry:
    """One reader or writer in the registry, and how to call it.

    Every reader is called as ``reader(fpath, **options)`` and every writer as
    ``writer(fpath, dataset, **options)``; the flags say where a format needs more
    than that.

    Attributes:
        function: The reader or writer function.
        options: Keyword arguments the function always gets for this extension,
            e.g. ``binary=False`` for formatted Plot3D.
        grid_file: True for split formats, whose grid is a file of its own; the
            function takes its path as the second argument.
        dtype: True for a writer that takes write_file's ``dtype`` (HDF5).
    """

    function: Callable[..., Any]
    options: dict[str, Any] = field(default_factory=dict)
    grid_file: bool = False
    dtype: bool = False


# To add a format: import its reader or writer above and add its extension below;
# read_file and write_file need no change.

# keys are lower-cased suffixes (including the dot)
_READERS: dict[str, _FormatEntry] = {
    # split binary: flow file + grid file (grid_file=...), several time steps (it=...)
    ".s8": _FormatEntry(read_binary_direct, grid_file=True),
    ".s4": _FormatEntry(read_binary_direct, grid_file=True),
    ".h5": _FormatEntry(read_hdf5),
    ".hdf5": _FormatEntry(read_hdf5),
    # Plot3D grids, binary or formatted (the reader tells them apart)
    ".x": _FormatEntry(read_plot3d),
    ".xyz": _FormatEntry(read_plot3d),
    ".p2dfmt": _FormatEntry(read_plot3d),
    ".p3dfmt": _FormatEntry(read_plot3d),
    ".q": _FormatEntry(read_plot3d_flow),
    ".dat": _FormatEntry(read_tecplot_ascii),
    # needs pytecplot (pip install cfd-io[tecplot]), imported when a file is read
    ".plt": _FormatEntry(read_tecplot_plt),
    ".vtu": _FormatEntry(read_vtu),
    ".cgns": _FormatEntry(read_cgns),
    ".su2": _FormatEntry(read_su2),
}

# keys are lower-cased suffixes (including the dot)
_WRITERS: dict[str, _FormatEntry] = {
    # split binary: flow file + grid file (grid_file=...); a grid alone needs no grid_file
    ".s8": _FormatEntry(write_binary_direct, grid_file=True),
    ".s4": _FormatEntry(write_binary_direct, grid_file=True),
    # HDF5 takes the precision (dtype=...)
    ".h5": _FormatEntry(write_hdf5, dtype=True),
    ".hdf5": _FormatEntry(write_hdf5, dtype=True),
    # Plot3D grids: binary for .x/.xyz, formatted (ASCII) for .p2dfmt/.p3dfmt
    ".x": _FormatEntry(write_plot3d),
    ".xyz": _FormatEntry(write_plot3d),
    ".p2dfmt": _FormatEntry(write_plot3d, options={"binary": False}),
    ".p3dfmt": _FormatEntry(write_plot3d, options={"binary": False}),
    ".dat": _FormatEntry(write_tecplot_ascii),
    # needs pytecplot (pip install cfd-io[tecplot]), imported when a file is written
    ".plt": _FormatEntry(write_tecplot_plt),
    ".su2": _FormatEntry(write_su2),
    # first-order elements; call write_gmsh directly for order 2
    ".msh": _FormatEntry(write_gmsh),
}


# --------------------------------------------------
# look up the reader or writer for a file path by its suffix
# --------------------------------------------------
def _resolve_format(fpath: Path, registry: dict[str, _FormatEntry], label: str) -> _FormatEntry:
    """Look up the reader or writer for *fpath* by its suffix.

    Args:
        fpath: File path whose extension is inspected.
        registry: Mapping of suffix -> registry entry (_READERS or _WRITERS).
        label: Human-readable label for error messages (e.g. "reader").

    Returns:
        The registry entry of the file's format.

    Raises:
        ValueError: If the extension is not in *registry*.
    """

    # get the suffix in lowercase
    suffix = fpath.suffix.lower()

    # look up the suffix in the registry
    if suffix not in registry:
        # assemble supported suffixes for the error message
        supported = ", ".join(sorted(registry.keys()))
        # raise an error: suffix was not recognized
        raise ValueError(f"no {label} for extension '{suffix}'; supported: {supported}")
    return registry[suffix]


# --------------------------------------------------
# public API to read data
# --------------------------------------------------
def read_file(
    fpath: str | Path,
    *,
    grid_file: str | Path | None = None,
    it: int = 1,
    orient: bool = True,
) -> Dataset:
    """Read a file, dispatching by extension.

    Args:
        fpath: Path to the primary data file (flow file for split formats).
        grid_file: Path to the grid file (required for split formats;
            ignored for HDF5).
        it: Timestep index, 1-based (split format only).
        orient: If True (default), the returned dataset is reoriented to
            the canonical layout (j=0 wall, +i streamwise) using
            `cfd_io.orient.canonicalize_dataset`.  Pass ``orient=False``
            to keep the on-disk index ordering.

    Returns:
        `Dataset` containing grid, flow, and metadata.

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If any required file is missing.
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    # resolve the reader from the file extension (set above in _READERS)
    entry = _resolve_format(fpath, _READERS, "reader")

    # check: a split format's flow file is nothing without its grid file
    if entry.grid_file and grid_file is None:
        raise ValueError("split format requires a separate grid file; pass grid_file='...'")

    # debug output for devs
    logger.info("read_file: %s  (reader=%s)", fpath, entry.function.__name__)

    # dispatch to the format-specific reader
    reader = entry.function
    if entry.grid_file:
        # split formats: the grid file, and the time step to read
        ds = reader(fpath, grid_file, it=it, **entry.options)
    else:
        # every other format: the file holds the whole dataset
        ds = reader(fpath, **entry.options)

    # post-read canonical orientation: j=0 wall, +i streamwise
    if orient:
        from cfd_io.orient import canonicalize_dataset

        ds = canonicalize_dataset(ds)

    return ds


# --------------------------------------------------
# public API to write data
# --------------------------------------------------
def write_file(
    fpath: str | Path,
    dataset: Dataset,
    *,
    grid_file: str | Path | None = None,
    dtype: str = "f",
) -> Path:
    """Write data, dispatching by extension.

    Args:
        fpath: Output path (flow file for split formats).
        dataset: Dataset to write.
        grid_file: Output path for the grid file (required for split
            formats; ignored for HDF5).
        dtype: NumPy dtype for HDF5 datasets (default ``"f"`` = float32).

    Returns:
        Path to the created output file (the primary file when split).

    Raises:
        ValueError: If the file extension is not recognized.
    """
    # ensure fpath is a Path object
    fpath = Path(fpath)

    # resolve the writer from the file extension (set above in _WRITERS)
    entry = _resolve_format(fpath, _WRITERS, "writer")

    # check: split formats write flow and grid to separate files, so a flow needs a grid
    # file (a grid alone goes to fpath)
    if entry.grid_file and grid_file is None and dataset.flow:
        raise ValueError("split format requires a separate grid file; pass grid_file='...'")

    # debug output for devs
    logger.info("write_file: %s  (writer=%s)", fpath, entry.function.__name__)

    # split formats: the writer returns both paths
    writer = entry.function
    if entry.grid_file:
        if grid_file is None:
            # grid-only dataset: use fpath as the grid file
            _flow_path, grid_path = writer(fpath, fpath, dataset)
            return grid_path
        flow_path, _grid_path = writer(fpath, grid_file, dataset)
        return flow_path

    # every other format: one file; the entry says which keyword arguments the writer gets
    writer_options = dict(entry.options)
    if entry.dtype:
        writer_options["dtype"] = dtype
    output_path = writer(fpath, dataset, **writer_options)

    return output_path


# --------------------------------------------------
# public API to convert between formats (convenience wrapper around read/write)
# --------------------------------------------------
def do_convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_grid: str | Path | None = None,
    output_grid: str | Path | None = None,
    it: int = 1,
    attrs: dict[str, Any] | None = None,
    dtype: str = "f",
) -> Path:
    """Read one format, write another -- convenience wrapper.

    Combines ``read_file`` and ``write_file`` into a single call.
    Formats are inferred from the file extensions. This is a pure format
    conversion -- use ``cfd_ops`` for any data transformation (transpose,
    cut, translate, ...).

    Args:
        input_path: Source file path (flow file for split formats).
        output_path: Destination file path.
        input_grid: Grid file for the source (split format input).
        output_grid: Grid file for the destination (split format output).
        it: Timestep index, 1-based (split format input).
        attrs: Extra metadata to merge into the output.  These override
            any attributes already present in the source file.
        dtype: NumPy dtype for HDF5 output.

    Returns:
        Path to the created output file.
    """
    # debug output for devs
    logger.info("convert: %s -> %s", input_path, output_path)

    output_path = Path(output_path)

    # auto-generate grid filename for split output if not provided
    output_entry = _WRITERS.get(output_path.suffix.lower())
    split_output = output_entry is not None and output_entry.grid_file
    if output_grid is None and split_output:
        output_grid = output_path.with_name(f"{output_path.stem}_grid{output_path.suffix}")
        logger.info("convert: auto grid-out -> %s", output_grid)

    # read the source file into a Dataset
    ds = read_file(
        input_path,
        grid_file=input_grid,
        it=it,
    )

    # check: a grid of a single point means the reader didn't find the grid's
    # dimensions; writing it would hide that the data is gone
    if isinstance(ds.grid, StructuredGrid) and ds.grid.x.size == 1:
        raise ValueError(
            f"{input_path} was read as a single grid point; the reader didn't find "
            "the grid's dimensions, so nothing was written"
        )

    # merge: file attrs as base, caller overrides on top
    if attrs:
        ds.attrs = {**ds.attrs, **attrs}

    # write the data to the destination file
    result = write_file(
        output_path,
        ds,
        grid_file=output_grid,
        dtype=dtype,
    )

    # debug output for devs
    logger.info("convert: done -> %s", result)

    return result
