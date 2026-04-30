"""File metadata extraction.

Provides ``FileInfo`` (a dataclass) and ``get_info()`` which returns
structured metadata for any supported CFD file format without loading
the full dataset into memory.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------
# dataclass: structured file metadata
# --------------------------------------------------
@dataclass
class FileInfo:
    """Structured metadata for a CFD data file."""

    nx: int = 0
    ny: int = 0
    nz: int = 0
    np: int = 0
    nt: int = 0
    var_names: list[str] = field(default_factory=list)
    timesteps: list[int] = field(default_factory=list)
    precision: str = ""
    format: str = ""
    attrs: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------
# public API: extract metadata from a file
# --------------------------------------------------
def get_info(fpath: str | Path) -> FileInfo:
    """Extract metadata from a CFD data file.

    Dispatches by file extension. For split formats (.s8/.s4), the
    companion .cd file is located automatically.

    Args:
        fpath: Path to the data file.

    Returns:
        A ``FileInfo`` dataclass with grid dimensions, variable count,
        timestep info, and format details.

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If a required companion file is missing.
    """

    # ensure we have a Path object and get the suffix
    fpath = Path(fpath)
    suffix = fpath.suffix.lower()

    # .cd companion header files for split binary formats
    if suffix == ".cd":
        return _info_from_header(fpath)

    # .s8 / .s4 binary (resolve companion .cd automatically)
    elif suffix in (".s8", ".s4"):
        fpath = fpath.with_suffix(".cd")
        if not fpath.exists():
            raise FileNotFoundError(f"no companion .cd file found for {fpath}")
        return _info_from_header(fpath)

    # HDF5
    elif suffix in (".h5", ".hdf5"):
        return _info_from_hdf5(fpath)

    # Plot3D grid
    elif suffix in (".x", ".xyz"):
        return _info_from_plot3d_grid(fpath)

    # Plot3D solution
    elif suffix == ".q":
        return _info_from_plot3d_flow(fpath)

    # Tecplot ASCII
    elif suffix == ".dat":
        return _info_from_tecplot_ascii(fpath)

    # Tecplot binary
    elif suffix == ".plt":
        return _info_from_tecplot_binary(fpath)

    else:
        raise ValueError(f"unsupported file type: {suffix}")


# --------------------------------------------------
# read header file for split binary formats (direct access)
# --------------------------------------------------
def _info_from_header(cd_path: Path) -> FileInfo:
    """Extract info from a .cd companion header."""
    from cfd_io.readers.fortran_binary_direct import read_header

    header = read_header(cd_path)
    return FileInfo(
        nx=header.nx,
        ny=header.ny,
        nz=header.nz,
        np=header.np,
        nt=header.nt,
        var_names=header.var_names or [],
        timesteps=header.timesteps or [],
        precision=header.precision,
        format="split",
    )


# --------------------------------------------------
# get info from hdf5 files without loading arrays
# --------------------------------------------------
def _info_from_hdf5(fpath: Path) -> FileInfo:
    """Extract info from an HDF5 file without loading arrays."""
    import h5py

    with h5py.File(fpath, "r") as fobj:
        nx, ny, nz = 0, 0, 0
        precision = ""

        # read root attributes when present
        attrs: dict[str, object] = {}
        for key, value in fobj.attrs.items():
            if hasattr(value, "item"):
                try:
                    attrs[key] = value.item()
                    continue
                except Exception:
                    pass
            attrs[key] = value

        # infer grid dimensions from first grid dataset
        # on-disk HDF5 layout is Fortran-order (nk, nj, ni) so we read
        # nx from the LAST axis and nz from the FIRST axis
        if "grid" in fobj and isinstance(fobj["grid"], h5py.Group):
            for name in fobj["grid"]:
                ds = fobj["grid"][name]
                if isinstance(ds, h5py.Dataset):
                    shape = ds.shape
                    if len(shape) == 1:
                        nx, ny, nz = shape[0], 0, 0
                    elif len(shape) == 2:
                        # disk (nj, ni)
                        nx, ny, nz = shape[1], shape[0], 0
                    elif len(shape) >= 3:
                        # disk (nk, nj, ni)
                        nx, ny, nz = shape[2], shape[1], shape[0]
                    else:
                        nx, ny, nz = 0, 0, 0
                    precision = str(ds.dtype)
                    break
        else:
            # fallback: infer dimensions from root-level x/y/z datasets
            for coord_name in ("x", "y", "z"):
                if coord_name in fobj and isinstance(fobj[coord_name], h5py.Dataset):
                    ds = fobj[coord_name]
                    shape = ds.shape
                    if len(shape) == 1:
                        nx, ny, nz = shape[0], 0, 0
                    elif len(shape) == 2:
                        # disk (nj, ni)
                        nx, ny, nz = shape[1], shape[0], 0
                    elif len(shape) >= 3:
                        # disk (nk, nj, ni)
                        nx, ny, nz = shape[2], shape[1], shape[0]
                    else:
                        nx, ny, nz = 0, 0, 0
                    precision = str(ds.dtype)
                    break

        # extract flow variable names and timestep info
        var_names: list[str] = []
        timesteps: list[int] = []

        if "flow" in fobj and isinstance(fobj["flow"], h5py.Group):
            flow_grp = fobj["flow"]

            # separate timestep sub-groups from flat datasets
            ts_groups = []
            flat_datasets = []
            for name in flow_grp:
                child = flow_grp[name]
                if isinstance(child, h5py.Group):
                    ts_groups.append(name)
                elif isinstance(child, h5py.Dataset):
                    flat_datasets.append(name)

            # multi-timestep layout
            if ts_groups:
                ts_groups.sort()
                parsed_timesteps: list[int] = []
                for token in ts_groups:
                    try:
                        parsed_timesteps.append(int(token))
                    except ValueError:
                        continue
                timesteps = parsed_timesteps
                # get var names from first timestep group
                first_ts = flow_grp[ts_groups[0]]
                var_names = [n for n in first_ts if isinstance(first_ts[n], h5py.Dataset)]

            # flat layout
            elif flat_datasets:
                var_names = flat_datasets
        else:
            # fallback: flat root-level datasets (x/y/z are coordinates, others are variables)
            coord_names = {"x", "y", "z"}
            root_datasets = [
                name for name in fobj
                if isinstance(fobj[name], h5py.Dataset)
            ]
            var_names = [name for name in root_datasets if name not in coord_names]

            if not precision and root_datasets:
                precision = str(fobj[root_datasets[0]].dtype)

        return FileInfo(
            nx=nx,
            ny=ny,
            nz=nz,
            np=len(var_names),
            nt=max(len(timesteps), 1) if var_names else 0,
            var_names=var_names,
            timesteps=timesteps,
            precision=precision,
            format="hdf5",
            attrs=attrs,
        )


# --------------------------------------------------
# get info from plot 3d grid data
# --------------------------------------------------
def _info_from_plot3d_grid(fpath: Path) -> FileInfo:
    """Extract info from a Plot3D grid file without loading coordinates."""
    from cfd_io.readers.plot3d import read_plot3d

    # read_plot3d is lightweight for grid-only; gets dims from header records
    ds = read_plot3d(fpath)
    nx, ny, nz = ds.grid.shape
    return FileInfo(
        nx=nx,
        ny=ny,
        nz=nz,
        np=0,
        nt=0,
        precision=str(ds.grid.x.dtype),
        format="plot3d",
    )


# --------------------------------------------------
# get info from plot3d solution data
# --------------------------------------------------
def _info_from_plot3d_flow(fpath: Path) -> FileInfo:
    """Extract info from a Plot3D .q solution file."""
    from cfd_io.readers.plot3d_flow import read_plot3d_flow

    ds = read_plot3d_flow(fpath)
    var_names = list(ds.flow.keys())
    if var_names:
        first = ds.flow[var_names[0]].data
        nx, ny, nz = first.shape
        precision = str(first.dtype)
    else:
        nx, ny, nz = 0, 0, 0
        precision = ""

    return FileInfo(
        nx=nx,
        ny=ny,
        nz=nz,
        np=len(var_names),
        nt=1,
        var_names=var_names,
        precision=precision,
        format="plot3d_flow",
    )


# --------------------------------------------------
# get info from tecplot ascii files
# --------------------------------------------------
def _info_from_tecplot_ascii(fpath: Path) -> FileInfo:
    """Extract info from a Tecplot ASCII file."""
    from cfd_io.readers.tecplot_ascii import read_tecplot_ascii

    ds = read_tecplot_ascii(fpath)
    var_names = list(ds.flow.keys())
    nx, ny, nz = ds.grid.shape
    return FileInfo(
        nx=nx,
        ny=ny,
        nz=nz,
        np=len(var_names),
        nt=1 if var_names else 0,
        var_names=var_names,
        precision=str(ds.grid.x.dtype),
        format="tecplot",
    )


# --------------------------------------------------
# get info from tecplot binary files (requires pytecplot)
# --------------------------------------------------
def _info_from_tecplot_binary(fpath: Path) -> FileInfo:
    """Extract info from a Tecplot binary file."""
    from cfd_io.readers.tecplot_binary import read_tecplot_plt

    ds = read_tecplot_plt(fpath)
    var_names = list(ds.flow.keys())
    nx, ny, nz = ds.grid.shape
    return FileInfo(
        nx=nx,
        ny=ny,
        nz=nz,
        np=len(var_names),
        nt=1 if var_names else 0,
        var_names=var_names,
        precision=str(ds.grid.x.dtype),
        format="tecplot_binary",
    )
