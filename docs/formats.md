# Supported Formats

## Format Table

| Format | Read | Write | Extension | Split |
|--------|------|-------|-----------|-------|
| HDF5 | yes | yes | `.h5`, `.hdf5` | no |
| Plot3D grid | yes | yes | `.x`, `.xyz` | no (grid only) |
| Plot3D solution | yes | -- | `.q` | no (flow only) |
| Tecplot ASCII | yes | yes | `.dat` | no |
| Tecplot binary | yes | yes | `.plt` | no (requires pytecplot) |
| Fortran unformatted binary | yes | yes | -- | low-level I/O |
| Raw binary (+ header) | yes | yes | `.s8` | yes -- requires grid file |

## HDF5

Grouped layout with `/grid` and `/flow` groups. Supports single and
multi-timestep layouts. Root-level attributes store scalar metadata
(Mach, Re, etc.).

## Plot3D

Standard NASA Plot3D structured grid format. Binary (Fortran unformatted)
and ASCII variants are auto-detected. 2-D and 3-D grids supported.

## Tecplot

ASCII `.dat` files in both POINT and BLOCK data packing. Binary `.plt`
files require the `pytecplot` package (`pip install cfd-io[tecplot]`).

## Raw Binary

Split formats store grid and flow data in separate headerless binary files.
A companion header file provides the grid dimensions, variable names, and timestep indices.