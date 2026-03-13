"""cfd-io: Convert CFD data between formats."""

from importlib.metadata import version

__version__ = version("cfd-io")

from cfd_io.convert_mod import do_convert, read_file, write_file
from cfd_io.dataset import Dataset, Field, Grid, StructuredGrid, UnstructuredGrid
from cfd_io.info_mod import FileInfo, get_info
from cfd_io.readers.fortran_binary_direct import (
    BinaryHeader,
    read_binary_direct,
    read_binary_direct_x_line,
    read_binary_direct_xy_plane,
    read_binary_direct_xyz_volume,
    read_header,
)
from cfd_io.readers.fortran_binary_sequential import FortranBinaryReader
from cfd_io.readers.hdf5 import read_hdf5
from cfd_io.readers.plot3d import read_plot3d
from cfd_io.readers.plot3d_flow import read_plot3d_flow
from cfd_io.readers.plot3d_flow_ascii import read_plot3d_flow_ascii
from cfd_io.readers.plot3d_flow_binary import read_plot3d_flow_binary
from cfd_io.readers.plot3d_grid_ascii import read_plot3d_grid_ascii
from cfd_io.readers.plot3d_grid_binary import read_plot3d_grid_binary
from cfd_io.readers.tecplot_ascii import read_tecplot_ascii
from cfd_io.readers.cgns import read_cgns
from cfd_io.readers.vtu import read_vtu
from cfd_io.writers.fortran_binary_sequential import FortranBinaryWriter
from cfd_io.writers.hdf5 import write_hdf5

# optional: tecplot binary (.plt) via pytecplot
try:
    from cfd_io.readers.tecplot_binary import read_tecplot_plt
    from cfd_io.writers.tecplot_binary import write_tecplot_plt
except ImportError:  # pytecplot not installed
    pass
from cfd_io.writers.fortran_binary_direct import write_binary_direct
from cfd_io.writers.plot3d import write_plot3d
from cfd_io.writers.plot3d_grid_ascii import write_plot3d_grid_ascii
from cfd_io.writers.plot3d_grid_binary import write_plot3d_grid_binary
from cfd_io.writers.tecplot_ascii import write_tecplot_ascii

__all__ = [
    "BinaryHeader",
    "Dataset",
    "Field",
    "FileInfo",
    "FortranBinaryReader",
    "FortranBinaryWriter",
    "Grid",
    "StructuredGrid",
    "UnstructuredGrid",
    "do_convert",
    "get_info",
    "read_binary_direct",
    "read_binary_direct_x_line",
    "read_cgns",
    "read_binary_direct_xy_plane",
    "read_binary_direct_xyz_volume",
    "read_file",
    "read_hdf5",
    "read_header",
    "read_plot3d",
    "read_plot3d_flow",
    "read_plot3d_flow_ascii",
    "read_plot3d_flow_binary",
    "read_plot3d_grid_ascii",
    "read_plot3d_grid_binary",
    "read_tecplot_ascii",
    "read_vtu",
    "write_binary_direct",
    "write_file",
    "write_hdf5",
    "write_plot3d",
    "write_plot3d_grid_ascii",
    "write_plot3d_grid_binary",
    "write_tecplot_ascii",
]

# conditionally export pytecplot-dependent functions
try:
    read_tecplot_plt  # noqa: B018
    write_tecplot_plt  # noqa: B018
    __all__ += ["read_tecplot_plt", "write_tecplot_plt"]
except NameError:
    pass
