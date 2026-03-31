"""Extension-based dispatchers for reading and writing CFD data.

Supported formats:

==========  ================  =====
Extension   Format            Split
==========  ================  =====
``.s8``     float64           yes -- requires ``grid_file``
``.s4``     float32           yes -- requires ``grid_file``
``.h5``     HDF5              no
``.hdf5``   HDF5              no
``.x``      Plot3D grid       no (grid only)
``.xyz``    Plot3D grid       no (grid only)
``.q``      Plot3D solution   no (flow only)
``.dat``    Tecplot ASCII     no
``.plt``    Tecplot binary    no (requires pytecplot)
==========  ================  =====
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

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# format registry
# --------------------------------------------------

# keys are lower-cased suffixes (including the dot)
_READER_REGISTRY: dict[str, str] = {
    ".s8": "split",
    ".s4": "split",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".x": "plot3d",
    ".xyz": "plot3d",
    ".q": "plot3d_flow",
    ".dat": "tecplot",
    ".plt": "tecplot_binary",
    ".vtu": "vtu",
    ".cgns": "cgns",
    ".su2": "su2",
}

_WRITER_REGISTRY: dict[str, str] = {
    ".s8": "split",
    ".s4": "split",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".x": "plot3d",
    ".xyz": "plot3d",
    ".dat": "tecplot",
    ".plt": "tecplot_binary",
    ".su2": "su2",
}

# --------------------------------------------------
# look up the format tag for a file path by its suffix
# --------------------------------------------------
def _resolve_format(fpath: Path, registry: dict[str, str], label: str) -> str:
    """Look up the format tag for *fpath* by its suffix.

    Args:
        fpath: File path whose extension is inspected.
        registry: Mapping of suffix -> format tag.
        label: Human-readable label for error messages (e.g. "reader").

    Returns:
        Format tag (e.g. ``"hdf5"``, ``"split"``).

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
        raise ValueError(
            f"no {label} for extension '{suffix}'; "
            f"supported: {supported}"
        )
    return registry[suffix]


# --------------------------------------------------
# public API to read data
# --------------------------------------------------
def read_file(
    fpath: str | Path,
    *,
    grid_file: str | Path | None = None,
    it: int = 1,
) -> Dataset:
    """Read a file, dispatching by extension.

    Args:
        fpath: Path to the primary data file (flow file for split formats).
        grid_file: Path to the grid file (required for split formats;
            ignored for HDF5).
        it: Timestep index, 1-based (split format only).

    Returns:
        `Dataset` containing grid, flow, and metadata.

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If any required file is missing.
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    # resolve the format for reader from the file extension (set above in _READER_REGISTRY)
    fmt = _resolve_format(fpath, _READER_REGISTRY, "reader")

    # debug output for devs
    logger.info("read_file: %s  (format=%s)", fpath, fmt)

    if fmt == "split":

        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.fortran_binary_direct import read_binary_direct

        if grid_file is None:
            raise ValueError(
                "split format requires a separate grid file; "
                "pass grid_file='...'"
            )

        return read_binary_direct(
            fpath=fpath,
            gpath=grid_file,
            it=it,
        )

    elif fmt == "hdf5":
        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.hdf5 import read_hdf5

        return read_hdf5(fpath)

    elif fmt == "plot3d":
        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.plot3d import read_plot3d

        return read_plot3d(fpath)

    elif fmt == "plot3d_flow":
        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.plot3d_flow import read_plot3d_flow

        return read_plot3d_flow(fpath)

    elif fmt == "tecplot":
        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.tecplot_ascii import read_tecplot_ascii

        return read_tecplot_ascii(fpath)

    elif fmt == "tecplot_binary":
        # lazy import of appropriate reader (deferred until we know it is needed to avoid circular import)
        from cfd_io.readers.tecplot_binary import read_tecplot_plt

        return read_tecplot_plt(fpath)

    elif fmt == "vtu":
        from cfd_io.readers.vtu import read_vtu

        return read_vtu(fpath)

    elif fmt == "cgns":
        from cfd_io.readers.cgns import read_cgns

        return read_cgns(fpath)

    elif fmt == "su2":
        from cfd_io.readers.su2 import read_su2

        return read_su2(fpath)

    else:
        raise ValueError(f"unsupported format: {fmt}")


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

    # resolve the format for writer from the file extension (set above in _WRITER_REGISTRY)
    fmt = _resolve_format(fpath, _WRITER_REGISTRY, "writer")

    # debug output for devs
    logger.info("write_file: %s  (format=%s)", fpath, fmt)

    if fmt == "split":

        # lazy import of appropriate writer
        from cfd_io.writers.fortran_binary_direct import write_binary_direct

        if grid_file is None:
            # grid-only dataset: use fpath as the grid file
            if not dataset.flow:
                _flow_path, grid_path = write_binary_direct(
                    fpath=fpath,
                    gpath=fpath,
                    dataset=dataset,
                )
                return grid_path
            raise ValueError(
                "split format requires a separate grid file; "
                "pass grid_file='...'"
            )

        flow_path, _grid_path = write_binary_direct(
            fpath=fpath,
            gpath=grid_file,
            dataset=dataset,
        )
        return flow_path

    elif fmt == "hdf5":

        # lazy import of appropriate writer
        from cfd_io.writers.hdf5 import write_hdf5

        return write_hdf5(fpath, dataset, dtype=dtype)

    elif fmt == "plot3d":

        # lazy import of appropriate writer
        from cfd_io.writers.plot3d import write_plot3d

        return write_plot3d(fpath, dataset)

    elif fmt == "tecplot":

        # lazy import of appropriate writer
        from cfd_io.writers.tecplot_ascii import write_tecplot_ascii

        return write_tecplot_ascii(fpath, dataset)

    elif fmt == "tecplot_binary":

        # lazy import of appropriate writer
        from cfd_io.writers.tecplot_binary import write_tecplot_plt

        write_tecplot_plt(fpath, dataset)
        return fpath

    elif fmt == "su2":

        # lazy import of appropriate writer
        from cfd_io.writers.su2 import write_su2

        return write_su2(fpath, dataset)

    else:
        raise ValueError(f"unsupported format: {fmt}")


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
    swap_ij: bool = False,
    attrs: dict[str, Any] | None = None,
    dtype: str = "f",
) -> Path:
    """Read one format, write another -- convenience wrapper.

    Combines ``read_file`` and ``write_file`` into a single call.
    Formats are inferred from the file extensions.

    Args:
        input_path: Source file path (flow file for split formats).
        output_path: Destination file path.
        input_grid: Grid file for the source (split format input).
        output_grid: Grid file for the destination (split format output).
        it: Timestep index, 1-based (split format input).
        swap_ij: If True, swap i and j axes before writing output.
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
    if output_grid is None and output_path.suffix.lower() in (".s8", ".s4"):
        output_grid = output_path.with_name(f"{output_path.stem}_grid{output_path.suffix}")
        logger.info("convert: auto grid-out -> %s", output_grid)

    # read the source file into a Dataset
    ds = read_file(
        input_path,
        grid_file=input_grid,
        it=it,
    )

    # swap i/j axes when requested
    if swap_ij:
        # swap axes in grid coordinates
        grid_out = StructuredGrid(
            np.swapaxes(ds.grid.x, 0, 1),
            np.swapaxes(ds.grid.y, 0, 1),
            np.swapaxes(ds.grid.z, 0, 1),
        )

        # swap axes in all flow variables
        flow_out: dict[str, Field] = {}
        for name, field in ds.flow.items():
            flow_out[name] = Field(
                data=np.swapaxes(field.data, 0, 1),
                association=field.association,
            )

        # rebuild dataset with swapped orientation
        ds = Dataset(
            grid=grid_out,
            flow=flow_out,
            attrs=dict(ds.attrs),
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
