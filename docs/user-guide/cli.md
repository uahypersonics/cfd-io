# CLI Usage

The `cfd-io` command-line tool provides two subcommands:

| Command | Description |
|---------|-------------|
| [`info`](#info) | Inspect file metadata without loading data |
| [`convert`](#convert) | Convert CFD data between file formats |

!!! note
    Options can be placed before or after positional arguments.

## `info`

Show metadata summary for a data file.

```
cfd-io info [OPTIONS] PATH
```

| Argument | Description |
|----------|-------------|
| `PATH` | Path to a data file (`.h5`, `.x`, `.dat`, `.plt`) |

### Examples

=== "HDF5"

    ```bash
    cfd-io info sample_flow.h5
    ```

    ```
    format:    hdf5
    grid:      51 x 900 x 1
    variables: 8
      names:   ['pres', 'temp', 'uvel', 'vvel', 'wvel', 'dens', 'mach', 'mu']
    timesteps: 1
    precision: float64
    ```

=== "Plot3D"

    ```bash
    cfd-io info sample_grid.x
    ```

    ```
    format:    plot3d_grid
    grid:      51 x 900 x 1
    variables: 3
      names:   ['x', 'y', 'z']
    timesteps: 1
    precision: float64
    ```

=== "Tecplot"

    ```bash
    cfd-io info sample_flow.dat
    ```

    ```
    format:    tecplot_ascii
    grid:      51 x 900 x 1
    variables: 8
      names:   ['x', 'y', 'pres', 'temp', 'uvel', 'vvel', 'wvel', 'dens']
    timesteps: 1
    precision: float64
    ```

## `convert`

Convert CFD data between formats. Format is auto-detected from file extensions.

```
cfd-io convert [OPTIONS] INPUT_PATH
```

| Argument / Option | Short | Description |
|-------------------|-------|-------------|
| `INPUT_PATH` | | Input file path |
| `--output PATH` | `-o` | Output file path (default: `output.h5`) |
| `--grid PATH` | `-g` | Input grid file (required for split formats) |
| `--output-grid PATH` | | Output grid file (required when writing split formats) |
| `--debug` | | Enable debug logging |
| `--mach FLOAT` | | Store Mach number as metadata |
| `--re FLOAT` | | Store unit Reynolds number as metadata |
| `--temp-inf FLOAT` | | Store freestream temperature as metadata |

!!! note "Split formats"
    Formats like `.s8` / `.s4` store grid and flow in separate files.
    Use `--grid` to provide the input grid file, and `--output-grid`
    when writing to a split format.

### Examples

=== "HDF5 to Tecplot"

    ```bash
    cfd-io convert sample_flow.h5 -o output.dat
    ```

=== "Tecplot to HDF5"

    ```bash
    cfd-io convert sample_flow.dat -o output.h5
    ```

=== "Plot3D to HDF5"

    ```bash
    cfd-io convert sample_grid.x -o output.h5 --mach 6.0
    ```

=== "Binary to HDF5"

    ```bash
    cfd-io convert flow.s8 -g grid.s8 -o output.h5
    ```
