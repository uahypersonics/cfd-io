# Python API Usage

All I/O functions are available directly from the top-level `cfd_io` package.

```python
from cfd_io import read_file, write_file, do_convert, get_info
```

## Inspect a File

`get_info` returns a `FileInfo` object with grid dimensions, variable names,
and other metadata — without loading the full data arrays.

```python
from cfd_io import get_info

info = get_info("sample_flow.h5")
print(info.format)     # "hdf5"
print(info.nx, info.ny, info.nz)
print(info.var_names)
print(info.precision)  # "float64"
```

## Read a File

`read_file` returns a tuple of `(grid, flow, attrs)`:

- **grid** — `dict[str, ndarray]` with coordinate arrays (`"x"`, `"y"`, `"z"`)
- **flow** — `dict[str, ndarray]` with flow variables (`"pres"`, `"temp"`, ...)
- **attrs** — `dict` with scalar metadata (variable names, timesteps, etc.)

```python
from cfd_io import read_file

grid, flow, attrs = read_file("sample_flow.h5")

# access coordinates
x = grid["x"]  # shape (nx, ny, nz)

# access flow variables
pressure = flow["pres"]
temperature = flow["temp"]
```

For split formats that store grid and flow separately:

```python
grid, flow, attrs = read_file("flow.s8", grid_file="grid.s8")
```

## Write a File

`write_file` takes the same `(grid, flow, attrs)` structure:

```python
from cfd_io import read_file, write_file

# read from one format
grid, flow, attrs = read_file("sample_flow.h5")

# write to another
write_file("output.dat", grid, flow, attrs)
```

## Convert Between Formats

`do_convert` combines `read_file` and `write_file` into a single call:

```python
from cfd_io import do_convert

# HDF5 to Tecplot
do_convert("sample_flow.h5", "output.dat")

# Plot3D to HDF5 with metadata
do_convert("sample_grid.x", "output.h5", attrs={"mach": 6.0})
```

For split formats, provide the grid files:

```python
do_convert(
    "flow.s8",
    "output.h5",
    input_grid="grid.s8",
)
```

## Working with the Data

Once you have the `grid` and `flow` dicts, use them with NumPy or any other tool:

```python
import numpy as np
from cfd_io import read_file

grid, flow, attrs = read_file("sample_flow.h5")

# compute velocity magnitude
u = flow["uvel"]
v = flow["vvel"]
vmag = np.sqrt(u**2 + v**2)

# slice a wall-normal profile at i=25
pres_profile = flow["pres"][25, :, 0]
y_profile = grid["y"][25, :, 0]
```
